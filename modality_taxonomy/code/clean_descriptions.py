"""
clean_descriptions.py — Remove degenerate tails from VLM-generated descriptions.

Some VLMs (especially Qwen2.5-VL-7B) degenerate into emoji spam, repeated
phrases, or repetitive tokens when generating long descriptions. This script
detects and truncates such tails while preserving the meaningful content.

Detection strategies:
  1. Emoji runs: 3+ consecutive emoji characters → truncate before the run
  2. Repeated words: same word 3+ times in a row → truncate before repetition
  3. Repeated phrases: a phrase of 2-6 words repeating 3+ times → truncate
  4. Repeated substrings: character-level repetition ("AgainAndAgainAndAgain")

Usage:
    # Clean a single file (overwrites in place)
    python clean_descriptions.py --input descriptions.json

    # Clean and save to new file
    python clean_descriptions.py --input descriptions.json --output descriptions_clean.json

    # Clean all files in a directory
    python clean_descriptions.py --input-dir results/1-describe/full_min100_max1024/

    # Dry run — show what would change without modifying files
    python clean_descriptions.py --input descriptions.json --dry-run

    # Verbose — show each truncation
    python clean_descriptions.py --input descriptions.json --verbose
"""

import argparse
import glob
import json
import os
import re
import unicodedata


def is_emoji(char):
    """Check if a character is an emoji."""
    cat = unicodedata.category(char)
    # Emoji are typically in 'So' (Symbol, other) category
    # but we also check common emoji ranges directly
    if cat == 'So':
        return True
    cp = ord(char)
    # Common emoji ranges
    return (0x1F600 <= cp <= 0x1F64F or   # emoticons
            0x1F300 <= cp <= 0x1F5FF or   # misc symbols
            0x1F680 <= cp <= 0x1F6FF or   # transport
            0x1F900 <= cp <= 0x1F9FF or   # supplemental
            0x1FA00 <= cp <= 0x1FA6F or   # chess symbols
            0x1FA70 <= cp <= 0x1FAFF or   # symbols extended
            0x2600 <= cp <= 0x26FF or     # misc symbols
            0x2700 <= cp <= 0x27BF or     # dingbats
            0xFE00 <= cp <= 0xFE0F or     # variation selectors
            0x200D == cp)                  # zero width joiner


def find_emoji_run(text, min_run=3):
    """Find the start of a run of min_run+ consecutive emoji characters.

    Returns the index in the text where the emoji run starts, or -1 if none found.
    Ignores isolated emoji (1-2) which are normal in VLM output.
    """
    emoji_count = 0
    run_start = -1

    i = 0
    while i < len(text):
        char = text[i]
        if is_emoji(char):
            if emoji_count == 0:
                run_start = i
            emoji_count += 1
        elif char in (' ', '\u200d', '\ufe0f', '\ufe0e'):
            # Spaces and ZWJ between emoji don't break the run
            if emoji_count > 0:
                pass  # continue the run
            else:
                emoji_count = 0
                run_start = -1
        else:
            if emoji_count >= min_run:
                return run_start
            emoji_count = 0
            run_start = -1
        i += 1

    if emoji_count >= min_run:
        return run_start
    return -1


def find_repeated_word(text, min_repeats=3):
    """Find where a single word repeats min_repeats+ times consecutively.

    Returns (start_index, word) or (-1, None).
    """
    words = text.split()
    for i in range(len(words)):
        word = words[i].strip('.,!?;:')
        if len(word) < 2:
            continue
        count = 1
        for j in range(i + 1, len(words)):
            if words[j].strip('.,!?;:').lower() == word.lower():
                count += 1
            else:
                break
        if count >= min_repeats:
            # Find the character position of word i in original text
            pos = 0
            for w_idx in range(i):
                pos = text.index(words[w_idx], pos) + len(words[w_idx])
            pos = text.index(words[i], pos)
            return pos, word
    return -1, None


def find_repeated_phrase(text, min_phrase_len=2, max_phrase_len=8, min_repeats=3):
    """Find where a phrase of N words repeats min_repeats+ times.

    Returns the start index of the first repetition, or -1.
    """
    words = text.split()
    best_start_word_idx = len(words)  # track earliest repetition found

    for phrase_len in range(min_phrase_len, min(max_phrase_len + 1, len(words) // min_repeats + 1)):
        for i in range(len(words) - phrase_len * min_repeats + 1):
            phrase = ' '.join(words[i:i + phrase_len]).lower()
            if len(phrase.strip()) < 4:
                continue
            count = 1
            pos = i + phrase_len
            while pos + phrase_len <= len(words):
                candidate = ' '.join(words[pos:pos + phrase_len]).lower()
                if candidate == phrase:
                    count += 1
                    pos += phrase_len
                else:
                    break
            if count >= min_repeats and i < best_start_word_idx:
                best_start_word_idx = i

    if best_start_word_idx < len(words):
        # Convert word index to character position
        pos = 0
        for w_idx in range(best_start_word_idx):
            pos = text.index(words[w_idx], pos) + len(words[w_idx])
        pos = text.index(words[best_start_word_idx], pos)
        return pos
    return -1


def find_repeated_substring(text, min_substr_len=5, min_repeats=3):
    """Find character-level repetitions like 'AgainAndAgainAndAgainAndAgain'.

    Looks for substrings of min_substr_len+ chars that repeat min_repeats+ times
    consecutively.

    Returns the start index of the repeated block, or -1.
    """
    # Only check the last portion of text (degeneration happens at the end)
    check_start = max(0, len(text) - 500)
    segment = text[check_start:]

    for substr_len in range(min_substr_len, min(50, len(segment) // min_repeats + 1)):
        for i in range(len(segment) - substr_len * min_repeats + 1):
            substr = segment[i:i + substr_len]
            if substr.strip() == '' or all(c in ' .,!?' for c in substr):
                continue
            count = 1
            pos = i + substr_len
            while pos + substr_len <= len(segment):
                if segment[pos:pos + substr_len] == substr:
                    count += 1
                    pos += substr_len
                else:
                    break
            if count >= min_repeats:
                return check_start + i
    return -1


def clean_description(text):
    """Clean a single description by removing degenerate tails.

    Returns (cleaned_text, was_modified, reason).
    """
    original = text

    # Strategy 1: Emoji runs (10+ consecutive Unicode emoji chars — flag emoji
    # like 🇺🇸 count as 2 chars each, so 10 accommodates ~5 visual emoji.
    # Degenerate spam like 😊×80 is always 20+ and easily caught.)
    emoji_start = find_emoji_run(text, min_run=10)
    if emoji_start > 0:
        text = text[:emoji_start].rstrip()
        # Try to end at the last sentence
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.5:  # only if we keep >50% of text
            text = text[:last_period + 1]
        if text != original:
            return text, True, f'emoji_run at char {emoji_start}'

    # Strategy 2: Repeated single word
    rep_start, rep_word = find_repeated_word(text, min_repeats=3)
    if rep_start > 0:
        text = text[:rep_start].rstrip()
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.5:
            text = text[:last_period + 1]
        if text != original:
            return text, True, f'repeated_word "{rep_word}" at char {rep_start}'

    # Strategy 3: Repeated phrase
    phrase_start = find_repeated_phrase(text, min_repeats=3)
    if phrase_start > 0:
        text = text[:phrase_start].rstrip()
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.5:
            text = text[:last_period + 1]
        if text != original:
            return text, True, f'repeated_phrase at char {phrase_start}'

    # Strategy 4: Hashtag spam (#UrbanGraffitiDetails #StreetGraffiti...)
    # Matches 3+ consecutive hashtags anywhere in text (not just at the end),
    # since degenerate text may follow the hashtags.
    hashtag_match = re.search(r'(#\w+[\s]*){3,}', text)
    if hashtag_match:
        text = text[:hashtag_match.start()].rstrip()
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.3:
            text = text[:last_period + 1]
        if text != original:
            return text, True, f'hashtag_spam at char {hashtag_match.start()}'

    # Strategy 5: Repeated substring
    substr_start = find_repeated_substring(text, min_repeats=3)
    if substr_start is not None and substr_start > 0:
        text = text[:substr_start].rstrip()
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > len(text) * 0.5:
            text = text[:last_period + 1]
        if text != original:
            return text, True, f'repeated_substring at char {substr_start}'

    # Strategy 6: Truncate at last complete sentence when text was cut mid-sentence
    # by max_tokens.  Only triggers if there is substantial content (>20 non-emoji
    # characters) after the last sentence-ending punctuation — does NOT strip
    # trailing emojis which are intentional text tokens (e.g. "...event. 🇺🇸🚀").
    stripped = text.rstrip()
    if stripped and stripped[-1] not in '.!?)\"':
        last_period = max(stripped.rfind('.'), stripped.rfind('!'), stripped.rfind('?'))
        if last_period > len(stripped) * 0.3:  # keep >30% of text
            tail = stripped[last_period + 1:]
            # Count non-emoji, non-whitespace chars in the tail
            tail_text_chars = sum(1 for c in tail if not is_emoji(c) and not c.isspace())
            if tail_text_chars > 20:  # substantial text after last period = mid-sentence truncation
                text = stripped[:last_period + 1]
                if text != original:
                    return text, True, f'incomplete_sentence (truncated at char {last_period + 1})'

    return text, False, None


def process_file(input_path, output_path=None, dry_run=False, verbose=False, backup=False):
    """Process a single descriptions JSON file.

    If backup=True, saves the original as *_raw.json before overwriting.

    Returns (n_total, n_modified, n_errors).
    """
    with open(input_path) as f:
        data = json.load(f)

    n_total = 0
    n_modified = 0
    n_errors = 0

    for img_id, entry in data.items():
        n_total += 1
        # Handle both flat format {"id": "text"} and nested {"id": {"text": "...", ...}}
        if isinstance(entry, str):
            text = entry
            is_nested = False
        elif isinstance(entry, dict):
            text = entry.get('text', '')
            is_nested = True
        else:
            n_errors += 1
            continue

        cleaned, was_modified, reason = clean_description(text)

        if was_modified:
            n_modified += 1
            old_words = len(text.split())
            new_words = len(cleaned.split())
            if verbose:
                print(f'  {img_id}: {old_words} → {new_words} words ({reason})')
                print(f'    OLD tail: ...{text[-80:]}')
                print(f'    NEW tail: ...{cleaned[-80:]}')

            if not dry_run:
                if is_nested:
                    data[img_id]['text'] = cleaned
                else:
                    data[img_id] = cleaned

    if output_path is None:
        output_path = input_path

    if not dry_run and n_modified > 0:
        # Save original as _raw.json before overwriting
        if backup:
            base, ext = os.path.splitext(input_path)
            raw_path = f'{base}_raw{ext}'
            if not os.path.exists(raw_path):
                import shutil
                shutil.copy2(input_path, raw_path)
                if verbose:
                    print(f'  Backup: {raw_path}')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    return n_total, n_modified, n_errors


def main():
    parser = argparse.ArgumentParser(description='Clean degenerate tails from VLM descriptions')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Single JSON file to process')
    group.add_argument('--input-dir', help='Directory containing JSON files to process')
    parser.add_argument('--output', default=None,
                        help='Output file (default: overwrite input). Only for --input mode.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without modifying files')
    parser.add_argument('--backup', action='store_true',
                        help='Save original as *_raw.json before overwriting with cleaned version')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show each truncation')

    args = parser.parse_args()

    if args.input:
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, 'generated_descriptions*.json')))
        if not files:
            print(f'No description files found in {args.input_dir}')
            return

    total_files = 0
    total_modified = 0

    for f in files:
        print(f'\n{"─"*60}')
        print(f'Processing: {f}')
        output = args.output if (args.input and args.output) else None
        n_total, n_modified, n_errors = process_file(
            f, output_path=output, dry_run=args.dry_run, verbose=args.verbose,
            backup=args.backup)
        print(f'  {n_total} descriptions, {n_modified} cleaned, {n_errors} errors')
        if args.dry_run and n_modified > 0:
            print(f'  (dry run — no files modified)')
        total_files += 1
        total_modified += n_modified

    print(f'\n{"═"*60}')
    print(f'Done: {total_files} files, {total_modified} descriptions cleaned')
    if args.dry_run:
        print('(dry run — no files were modified)')


if __name__ == '__main__':
    main()