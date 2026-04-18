# test_code.py
import os
import json
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from datas import get_dataset
from utils import get_config, set_seed
from gating import Gating
from uconn_io import load_u_conn
from tqdm import tqdm
from datetime import datetime
import pandas as pd

set_seed(9728)


def get_dtype(dtype: str):
    return getattr(torch, dtype)

def get_block_modules(model) -> list:
    architectures = model.config.architectures[0]
    if "Gemma" in architectures:
        return list(model.language_model.layers)
    elif "Mistral" in architectures:
        return list(model.model.layers)
    elif "Llama" in architectures:
        return list(model.model.layers)
    elif "Phi" in architectures:
        return list(model.model.layers)
    else:
        raise ValueError(f"(get_block_modules) Unsupported model: {architectures}")

def extract_after_attention(target_layer):
    if hasattr(target_layer, "post_attention_layernorm"):
        return target_layer.post_attention_layernorm
    elif hasattr(target_layer, "post_feedforward_layernorm"):
        return target_layer.post_feedforward_layernorm
    else:
        raise ValueError(f"(extract_after_attention) Unsupported target layer: {target_layer}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract u_conn activation (attn_out, minimal)")
    p.add_argument("-m", "--model", type=str, default="google/gemma-3-4b-it")
    p.add_argument("-t", "--token", type=str)
    p.add_argument("--u-conn-path", type=str, required=True, help="Path to u_conn safetensors")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-dir", type=str, default="./result/gemma/")
    p.add_argument("--layer-index", type=int, default=33)
    p.add_argument("--alpha", type=float, default=0.5, help="Scale for u_conn")
    p.add_argument("--only-logic-connective", action="store_true")
    p.add_argument("--inital-search", type=int, default=0)
    p.add_argument("--type", type=str, default="add")
    p.add_argument("--my-name", type=str, default="testing")

    # Default
    p.add_argument("--positions", type=str, default="-1", help="Comma-separated positions (e.g., '-1' or '0,-1')")
    p.add_argument("--max-new-tokens", type=int, default=15000)
    p.add_argument("--dataset-name", type=str, default="zebra_logic")
    p.add_argument("--token-search-window", type=int, default=4)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--is-test", action="store_true", help="Is test mode")
    return p.parse_args()

def check_logical_connective(tokenizer, logic_check_store, logic_connective_set):
    for i in range(1, len(logic_check_store) + 1):
        cur = tokenizer.decode(logic_check_store[i])
        if cur in logic_connective_set:
            return True, cur, i
    return False, None, None

def store_logic_check(logic_check_store, next_token):
    for i in range(1, len(logic_check_store) + 1):
        logic_check_store[i].append(next_token)


def load_dataset(dataset_name: str, tokenizer: AutoTokenizer, args: argparse.Namespace):
    dataset_config = get_config(dataset_name)
    dataset_module = get_dataset(dataset_name)(dataset_config, tokenizer, is_test=args.is_test)
    dataset = dataset_module.load()
    return dataset


def steering_injector(model, u_pos, args):
    blocks = get_block_modules(model)
    target_layer = blocks[args.layer_index]
    after_attention = extract_after_attention(target_layer)
    
    holder = {"is_injected": False}
    

    def hook_fn(module, inputs):
        x = inputs[0]
        if not holder["is_injected"]:
            return x
        else:
            if args.type == "add":
                steering_vector = u_pos * args.alpha
            else:
                raise ValueError(f"Invalid task: {args.type}")
            holder["is_injected"] = False
            return x + steering_vector

    handle = after_attention.register_forward_pre_hook(hook_fn)
    return handle, holder


def get_config_dict(args):
    return {
    "run_info": {
        "my_name": args.my_name,
        "save_dir": args.save_dir,
        "save_steps": args.save_steps,
    },
    "model_config": {
        "model": args.model,
    },
    "data_config": {
        "dataset_name": args.dataset_name,
        "num_samples": args.num_samples,
        "is_test": args.is_test,
    },
    "generation_config": {
        "max_new_tokens": args.max_new_tokens,
        "positions": args.positions,
        "token_search_window": args.token_search_window,
    },
    "steering_config": {
        "type": args.type,
        "u_conn_path": args.u_conn_path,
        "layer_index": args.layer_index,
        "alpha": args.alpha,
        "inital_search": args.inital_search,
        "only_logic_connective": args.only_logic_connective,
    },
    "hardware_config": {
        "device": args.device,
        "dtype": args.dtype,
    }
}

def _pick_connective_or_argmax(logits: torch.Tensor, gating, only_logic_connective: bool):
    if only_logic_connective:
        idx = getattr(gating, "_conn_idx_tensor", None)
        if idx is not None and idx.numel() > 0:
            V = logits.shape[-1]
            idx = idx[idx < V].to(logits.device)
            if idx.numel() > 0:
                sub = logits.index_select(dim=-1, index=idx)
                rel = torch.argmax(sub, dim=-1)
                sel = idx[rel.squeeze(0)]
                if sel.ndim == 0:
                    sel = sel.unsqueeze(0)
                return sel
    return torch.argmax(logits, dim=-1)

def main(args):
    args_json = vars(args)
    print("-=-=-=-=- Current args -=-=-=-=-")
    print(args_json)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=get_dtype(args.dtype), token=args.token).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    u_pos = load_u_conn(args.u_conn_path).to(torch.bfloat16).to(args.device)
    handle, holder = steering_injector(model, u_pos, args)
    gating = Gating(args, tokenizer)    
    dataset = load_dataset(args.dataset_name, tokenizer, args)

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_" + args.my_name

    eos_list = [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]] # 106 for gemma
    if "gemma" in args.model:
        eos_list.append(106)
    if "Phi" in args.model or "phi" in args.model:
        eos_list.append(200020)

    save_dir = os.path.join(args.save_dir, now_time)
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(args_json, f, indent=4, ensure_ascii=False)
    print(f"Configuration saved to: {config_path}")

    results = []

    num_samples = min(args.num_samples, len(dataset))
    for i in tqdm(range(num_samples)):
        prompt = dataset[i]["prompt"]
        answer = dataset[i]["answer"]
        steer_count = 0
        steer_positions = []
        

        if "tag" in dataset[i].keys():
            tag = dataset[i]["tag"]
        else:
            tag = "no tag"
        
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model(**input_ids, use_cache=True)
            next_token = output.logits[:, -1, :].argmax(dim=-1)
            past_key_value = output.past_key_values
            gating.add_store(
                next_token.item(), 
                output.past_key_values,
                output.logits[:, -1, :].to("cpu")
            )
            n = 1
            while n < args.max_new_tokens:
                output = model(
                    next_token.unsqueeze(-1),
                    past_key_values=past_key_value,
                    use_cache=True
                )
                next_token = output.logits[:, -1, :].argmax(dim=-1)

                gating.add_store(
                    next_token.item(), 
                    output.past_key_values,
                    output.logits[:, -1, :].to("cpu")
                )
                past_key_value = output.past_key_values
                n += 1
                
                if next_token.item() in eos_list:
                    break
                
                if args.alpha == 0.0:
                    continue

                is_logical_connective, cur, index = gating.check_logical_connective()
                if is_logical_connective and n > args.inital_search:
                    ok, metrics = gating.passes_all(index)
                    if not ok:
                        continue

                    steer_count += 1
                    steer_positions.append(n - index)

                    prev_token_id = gating.main_store[-index-1]
                    prev_past     = gating.past_key_value_store[-index-1]

                    holder["is_injected"] = True
                    prev_token = torch.tensor([[prev_token_id]], device=next_token.device)
                    output = model(input_ids=prev_token,
                                   past_key_values=prev_past,
                                   use_cache=True)
                    next_token = _pick_connective_or_argmax(output.logits[:, -1, :], gating, args.only_logic_connective)
                    
                    gating.clear_store(index)
                    gating.add_store(
                        next_token.item(), 
                        output.past_key_values,
                        output.logits[:, -1, :].to("cpu")
                    )
                    past_key_value = output.past_key_values

                    for _ in range(index + args.token_search_window + 1):
                        output = model(
                            next_token.unsqueeze(-1),
                            past_key_values=past_key_value,
                            use_cache=True
                        )
                        next_token = output.logits[:, -1, :].argmax(dim=-1)
                        gating.add_store(
                            next_token.item(), 
                            output.past_key_values,
                            output.logits[:, -1, :].to("cpu")
                        )
                        past_key_value = output.past_key_values
                    n += args.token_search_window + 1

        
        results.append({
            "prompt": prompt,
            "answer": answer,
            "generated": tokenizer.decode(gating.main_store),
            "steer_positions": steer_positions,
            "tag": tag,
        })
        gating.clear()

        if (i + 1) % args.save_steps == 0:
            df = pd.DataFrame(results)
            file_name = args.model.replace('/', '_') + f"_{i}.parquet"
            cur_dir = os.path.join(save_dir, file_name)
            df.to_parquet(cur_dir)

    # save as parquet
    df = pd.DataFrame(results)
    file_name = args.model.replace('/', '_') + f"_{i}.parquet"
    cur_dir = os.path.join(save_dir, file_name)
    df.to_parquet(cur_dir)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
