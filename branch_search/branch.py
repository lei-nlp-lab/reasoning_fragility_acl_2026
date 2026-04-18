from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import math
import copy
import pandas as pd
from datas import get_dataset
from datetime import datetime
from utils import get_config, set_seed
from tqdm import tqdm
from gating import Gating
import os
import json
import gc

set_seed(9728)

def _entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    logp = torch.log(p + 1e-12)
    return float((-(p * logp).sum(dim=-1)).item())

def _confidence_from_logits(logits: torch.Tensor, k=50) -> float:
    logp = torch.log_softmax(logits, dim=-1)
    topk_logps, _ = torch.topk(logp, k=k)
    return float(-topk_logps.mean().item())

def free_cuda(is_free: bool = False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--task", type=str, default="zebra_logic")
    parser.add_argument("--my_name", type=str, default="branch")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--k-top", type=int, default=20)
    parser.add_argument("--n-lookahead", type=int, default=20)
    parser.add_argument("--mx-token", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=15000)
    parser.add_argument("--save-dir", type=str, default="./runs")
    parser.add_argument("--save-step", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--entropy-gate", type=float, default=1000.0)
    parser.add_argument("--conf-topk", type=int, default=50)
    parser.add_argument("--conf-gate", type=float, default=1000.0)
    parser.add_argument("--end-branch-index", type=int, default=15000)
    parser.add_argument("--is-test", action="store_true")
    parser.add_argument("--start-branch-index", type=int, default=10)
    return parser.parse_args()


def branch_manager(
    args,
    model,
    tokenizer,
    past_key_values_at_branch,
    candidate_tokens,
    eos_list,
):
    L = args.n_lookahead
    eps = 1e-8

    if candidate_tokens is None or len(candidate_tokens) == 0:
        raise ValueError("candidate_tokens must be a non-empty list.")

    if len(candidate_tokens) < 2:
        tok = candidate_tokens[0]
        tok_id = int(tok[0] if isinstance(tok, (list, tuple)) else tok)
        next_input = torch.tensor([[tok_id]], dtype=torch.long, device=model.device)
        return {
            "tokens": [tok_id],
            "eos": tok_id in eos_list,
            "entropy": None,
            "confidence": None,
            "past_key_value": past_key_values_at_branch,
            "next_token": next_input,
        }

    branch_info = []
    H_list = []
    S_list = []

    for cand in candidate_tokens:
        cand_seq = list(cand) if isinstance(cand, (list, tuple)) else [int(cand)]
        tokens = cand_seq.copy()

        past = past_key_values_at_branch

        with torch.no_grad():
            output = None
            for tok in cand_seq:
                inp = torch.tensor([[tok]], dtype=torch.long, device=model.device)
                output = model(inp, past_key_values=past, use_cache=True)
                past = output.past_key_values

            entropy_sum = 0.0
            logprob_sum = 0.0
            steps = 0
            eos = False
            next_input = None

            for _ in range(L):
                logits = output.logits[:, -1, :]

                next_token = logits.argmax(dim=-1)

                entropy = _entropy_from_logits(logits.detach().cpu())

                log_probs = torch.log_softmax(logits, dim=-1)
                lp = log_probs.gather(-1, next_token.unsqueeze(-1)).item()

                entropy_sum += float(entropy)
                logprob_sum += float(lp)
                steps += 1

                tokens.append(next_token.item())
                next_input = next_token.unsqueeze(-1)  # shape [1,1]

                if next_token.item() in eos_list:
                    eos = True
                    break

                output = model(next_input, past_key_values=past, use_cache=True)
                past = output.past_key_values

        H = entropy_sum / max(steps, 1)
        S = logprob_sum / max(steps, 1)

        H_list.append(H)
        S_list.append(S)

        branch_info.append({
            "tokens": tokens,
            "eos": eos,
            "entropy": H,
            "confidence": S,
            "past_key_value": past,
            "next_token": next_input,
        })

    mu_H = sum(H_list) / len(H_list)
    mu_S = sum(S_list) / len(S_list)

    var_H = sum((h - mu_H) ** 2 for h in H_list) / len(H_list)
    var_S = sum((s - mu_S) ** 2 for s in S_list) / len(S_list)
    sigma_H = math.sqrt(var_H) if var_H > 0 else 1.0
    sigma_S = math.sqrt(var_S) if var_S > 0 else 1.0

    scores = []
    for H, S in zip(H_list, S_list):
        H_tilde = (H - mu_H) / (sigma_H + eps)
        S_tilde = (S - mu_S) / (sigma_S + eps)
        scores.append(H_tilde - S_tilde)

    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    branch_info[best_idx]["score"] = scores[best_idx]
    return branch_info[best_idx]


def main(args: argparse.Namespace):
    model_name = args.model
    task = args.task
    my_name = args.my_name
    dtype = args.dtype
    k_top = args.k_top
    n_lookahead = args.n_lookahead
    mx_token = args.mx_token
    max_new_tokens = args.max_new_tokens
    save_dir = args.save_dir
    save_step = args.save_step
    num_samples = args.num_samples
    entropy_gate = args.entropy_gate
    conf_topk = args.conf_topk
    conf_gate = args.conf_gate
    end_branch_index = args.end_branch_index
    is_test = args.is_test
    start_branch_index = args.start_branch_index

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()
    dataset_config = get_config(task)
    dataset = get_dataset(task)(dataset_config, tokenizer, is_test=is_test)

    eos_list = [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.encode("<end_of_turn>")[1]]
    if "Phi" in model_name or "phi" in my_name:
        eos_list.append(200020)
    if "gemma" in model_name or "gem" in my_name:
        eos_list.append(106)

    save_path = os.path.join(save_dir, my_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_path, exist_ok=True)

    # save args
    args_path = os.path.join(save_path, "args.json")
    with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    gating = Gating(args, tokenizer)
    logic_idx = gating._conn_idx_tensor

    results = []
    num_samples = min(num_samples, len(dataset))
    for i in tqdm(range(num_samples)):
        st_info = []
        prompt = dataset[i]["prompt"]
        answer = dataset[i]["answer"]
        if "tag" in dataset[i].keys():
            tag = dataset[i]["tag"]
        else:
            tag = "no tag"

        with torch.no_grad():

            gating = Gating(args, tokenizer)

            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model(**input_ids, use_cache=True)
            next_token = output.logits[:, -1, :].argmax(dim=-1)
            past_key_value = output.past_key_values

            gating.add_store(
                next_token.item(), 
                past_key_value,
                output.logits[:, -1, :].to("cpu")
            )
            
            n = 1
            while n < max_new_tokens:
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

                if next_token.item() in eos_list:
                    break

                n += 1

                is_logical_connective, cur, index = gating.check_logical_connective()

                if n > end_branch_index or n < start_branch_index:
                    continue

                if is_logical_connective:
                    softmax_at_branch_point = gating.softmax_store[-index]

                    all_connective_tokens = gating._conn_idx_tensor.to(softmax_at_branch_point.device)
                    all_connective_softmax = torch.index_select(softmax_at_branch_point, 0, logic_idx)
                    
                    mask = all_connective_softmax > 0.01
                    filtered_softmax = all_connective_softmax[mask]
                    filtered_tokens = all_connective_tokens[mask]

                    k_for_topk = min(k_top, len(filtered_tokens))
                    
                    if k_for_topk > 1:
                        topk_logps, topk_indices = torch.topk(filtered_softmax, k=k_for_topk)
                        topk_token_ids = filtered_tokens[topk_indices].tolist()
                    else:
                        continue
                    
                    original_connective_first_token = gating.main_store[-index]
                    candidate_tokens = [original_connective_first_token] + topk_token_ids
                    candidate_tokens = list(dict.fromkeys(candidate_tokens))


                    past_key_values_at_branch = gating.past_key_value_store[-index]
                        
                    branch_out = branch_manager(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        past_key_values_at_branch=past_key_values_at_branch,
                        # base_prompt=prompt + tokenizer.decode(gating.main_store),
                        candidate_tokens=candidate_tokens,
                        eos_list=eos_list,
                    )

                    gating.clear_store(index)

                    tokens = branch_out["tokens"]
                    eos = branch_out["eos"]
                    past_key_value = branch_out["past_key_value"]
                    next_token = torch.tensor([tokens[-1]], device=model.device, dtype=torch.long)
                    
                    st_info.append({
                        "from_token": cur,
                        "to_token": tokenizer.decode(tokens[0]),
                        "position": n - index,
                    })
                    n -= index
                    n += n_lookahead
                    gating.add_main_store(tokens)
                    
                    
        results.append({
            "prompt": prompt,
            "answer": answer,
            "generated": tokenizer.decode(gating.main_store),
            "steer_info": st_info,
            "tag": tag,
        })
        free_cuda(is_free=True)

        if (i + 1) % save_step == 0:
            cur_save_path = os.path.join(save_path, f"{my_name}_{i}.parquet")
            df = pd.DataFrame(results)
            df.to_parquet(cur_save_path)
    
    cur_save_path = os.path.join(save_path, f"{my_name}_{i}.parquet")
    df = pd.DataFrame(results)
    df.to_parquet(cur_save_path)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
