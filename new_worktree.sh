#!/usr/bin/env bash
set -euo pipefail                                          # stop on errors, unset vars, and failed pipes

REPO="$HOME/mh-neuron"                                     # main repo root
NAME="enrichment_analysis"                                 # this section's name
WT="$REPO/myrepo-${NAME}"                                  # worktree dir: ~/mh-neuron/myrepo-enrichment_analysis
BR="session/${NAME}"                                       # branch: session/enrichment_analysis
SESSION="claude"                                           # existing tmux session to add a window to

cd "$REPO"                                                 # operate from the main repo

if [ ! -d "$WT" ]; then                                    # only create the worktree if it doesn't exist yet
  git worktree add "$WT" -b "$BR"                          # new worktree on a new branch
fi
git worktree list                                          # confirm it's registered before tmux points at it

cd "$WT"                                                   # move into the new worktree
uv sync                                                    # build THIS worktree's own .venv from pyproject.toml

if [ ! -f CLAUDE.local.md ]; then                          # per-session scope file (gitignored); don't clobber if present
  cat > CLAUDE.local.md << 'EOF'
# Session scope: enrichment_analysis
Working branch: session/enrichment_analysis
Task: <fill in the specific goal for this session>
Out of scope: <anything this session must not touch>
EOF
fi

[ -f .claude/settings.local.json ] || echo '{}' > .claude/settings.local.json
                                                           # seed the local permissions layer if missing (empty object)
grep -n 'settings.local.json' .gitignore || \
  echo '**/.claude/settings.local.json' >> .gitignore     # ensure the local file is never committed

if ! tmux has-session -t "$SESSION" 2>/dev/null; then      # if the 'claude' session doesn't exist, create it...
  tmux new-session -d -s "$SESSION" -n "$NAME" -c "$WT"     # ...detached, first window points at this worktree
else
  tmux new-window  -t "$SESSION" -n "$NAME" -c "$WT"        # otherwise add a new window for this worktree
fi

tmux send-keys -t "${SESSION}:${NAME}" 'source .venv/bin/activate' Enter   # activate this worktree's venv (torch available)
tmux send-keys -t "${SESSION}:${NAME}" 'claude' Enter                       # launch Claude Code in that window

tmux attach -t "$SESSION"                                  # connect to the sessionq:

