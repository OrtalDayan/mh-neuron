import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import pdb
FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    # structs = [dataset.build_prompt(data.iloc[i]) for i in range(lt)]

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'
    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res
import torch

def modify_params(model, layer_num, module_type):
    # 获取对应层的参数路径
    layer_prefix = f"language_model.model.layers.{layer_num}"
    
    # 定义模块名和对应的参数
    module_params = {
        'self_attn': [
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
            f"{layer_prefix}.self_attn.o_proj.weight"
        ],
        'mlp': [
            f"{layer_prefix}.mlp.gate_proj.weight",
            f"{layer_prefix}.mlp.up_proj.weight",
            f"{layer_prefix}.mlp.down_proj.weight"
        ],
        # 'self_attn': [
            
        #     f"{layer_prefix}.self_attn.v_proj.weight",
           
        # ],
        # 'mlp': [
        #     f"{layer_prefix}.mlp.down_proj.weight"
        # ]
    }
    
    # 获取模块的参数
    params_to_modify = module_params.get(module_type)
    if params_to_modify is None:
        raise ValueError("Invalid module type. Choose 'self_attn' or 'mlp'.")
    
    # 遍历并修改参数
    for param_name in params_to_modify:
        if param_name in model.model.state_dict():
            param = model.model.state_dict()[param_name]
            pdb.set_trace()
            new_param = torch.ones_like(param)  # 创建一个新的参数，形状和原来一致，值都为1
            len=param.size()[0]
            new_param=new_param/len
            model.model.state_dict()[param_name].copy_(new_param)  # 更新模型参数
        else:
            print(f"Warning: {param_name} not found in the model state_dict.")
    
    return model


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4,merge_model=None,cut_layer=None,cut_module=None):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    
    if cut_layer!=None and cut_module!=None:
        model = modify_params(model, cut_layer, cut_module)

    if merge_model: 
        if 'llava' in model_name or 'intern' in model_name.lower() or 'Qwen' in model_name:
            # Load the saved state_dict
            # 
            # merged_state_dict = torch.load(merge_model)
            # model.model.language_model.load_state_dict(merged_state_dict)
            # Print the device information for each parameter in the original model
            #  Print the device information for each parameter in the original model
            device_map = {}
            for name, param in model.model.language_model.named_parameters():
                device_map[name] = param.device  # Record the device for each parameter

            # Load the merged model's state dictionary directly to CUDA
            merged_state_dict = torch.load(merge_model, map_location='cpu')

            # Reconcile key namespace: merge_pmbt.py saves the full VLM state_dict, so
            # language-model params carry a wrapper prefix that the load target here (the
            # bare text model, e.g. LlamaModel / Qwen2_5_VLTextModel) lacks. The prefix
            # varies by architecture: "model." (LLaVA/InternVL) or "language_model."
            # (Qwen2.5-VL). Detect it from a sample target key, strip it, and rely on
            # strict=False to drop unmatched keys (lm_head.weight, visual.* tower, ...).
            sample_target_key = next(iter(device_map))
            strip_prefix = ''
            for _cand in ('', 'model.', 'language_model.', 'model.language_model.'):
                if (_cand + sample_target_key) in merged_state_dict:
                    strip_prefix = _cand
                    break

            # Keep tensors on CPU here — load_state_dict copies CPU→GPU in-place into
            # the existing param buffers, avoiding a 13 GB duplicate on-device that would
            # OOM a 20 GB GPU. (Pre-moving to GPU as the original code did doubled memory.)
            new_state_dict = {}
            skipped = []
            mlp_matched = 0
            for name, param in merged_state_dict.items():
                key = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
                if key in device_map:
                    new_state_dict[key] = param  # CPU tensor; load_state_dict handles device
                    if any(p in key for p in ('gate_proj', 'up_proj', 'down_proj')):
                        mlp_matched += 1
                    # InternLM2 (used by InternVL2.5) uses feed_forward.w1/w2/w3
                    # instead of the LLaMA-style gate/up/down_proj naming.
                    elif 'feed_forward' in key and any(p in key for p in ('.w1.', '.w2.', '.w3.')):
                        mlp_matched += 1
                else:
                    skipped.append(name)
            print(f'[merge_model] strip_prefix={strip_prefix!r}; matched {len(new_state_dict)} '
                  f'keys into language_model ({mlp_matched} of them MLP gate/up/down).')
            if skipped:
                print(f'[merge_model] Skipping {len(skipped)} keys not in language_model '
                      f'(e.g. {skipped[:3]}). strict=False will ignore them.')
            # Loud guard: zero matched MLP weights means the prefix contract broke and the
            # merge silently no-opped — the eval would just re-measure base LLaVA-1.5.
            # Abort rather than report misleading numbers.
            if mlp_matched == 0:
                raise RuntimeError(
                    f'[merge_model] FATAL: 0 MLP weights matched the load target from '
                    f'{merge_model}. Merged checkpoint did NOT take. '
                    f'sample .pth key={next(iter(merged_state_dict))!r}, '
                    f'sample target key={sample_target_key!r}.')

            # Load the new state dictionary into the language_model
            missing, unexpected = model.model.language_model.load_state_dict(
                new_state_dict, strict=False)
            if missing:
                print(f'[merge_model] {len(missing)} missing keys (kept from base VLM): '
                      f'{missing[:3]}{"..." if len(missing) > 3 else ""}')
            if unexpected:
                print(f'[merge_model] {len(unexpected)} unexpected keys: '
                      f'{unexpected[:3]}{"..." if len(unexpected) > 3 else ""}')

            # Optionally, release any unnecessary memory
            del merged_state_dict  # Release the original merged state dict from memory
            torch.cuda.empty_cache()  # Clear CUDA cache
        elif 'idefics' in model_name:
            merged_state_dict = torch.load(merge_model)
            # 
            model.model.model.text_model.load_state_dict(merged_state_dict)
    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])
        # 
        response = model.generate(message=struct, dataset=dataset_name)
        # response = model.generate(message=struct, dataset=dataset_name,return_dict_in_generate=True, output_scores=True)
        # 
        # print('hah')
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False,merge_model=None,cut_layer=None,cut_module=None):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc,merge_model=merge_model,cut_layer=cut_layer,cut_module=cut_module)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
