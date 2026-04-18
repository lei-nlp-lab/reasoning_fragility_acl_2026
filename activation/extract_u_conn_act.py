# extract_u_conn_act.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Set, Dict, Optional
import torch
import os
import random
from safetensors.torch import save_file
import torch.nn.functional as F
from connective_ids import build_connective_first_token_set
from datetime import datetime


def get_dtype(dtype: str):
    return getattr(torch, dtype)

def _autocast_ctx(args):
    dev = getattr(args, "device", "cuda")
    dt  = get_dtype(args.dtype)
    return torch.autocast(device_type="cuda", dtype=dt, enabled=str(dev).startswith("cuda"))


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

def logical_connective_position(
    text: str,
    tokenizer: AutoTokenizer,
    logical_connective_set: set,
    token_search_window: int = 4,
) -> List[Tuple[int, int]]:
    positions = []
    encoded_text = tokenizer.encode(text)
    for i in range(len(encoded_text)):
        for span_size in range(token_search_window, 0, -1):
            start_index = max(0, i - span_size)
            sub_text = encoded_text[start_index:i]
            if len(sub_text) == 0:
                continue
            first_token_index = sub_text[0]
            token_text = tokenizer.decode(sub_text)
            if token_text in logical_connective_set:
                positions.append({
                    "prefix_text": tokenizer.decode(encoded_text[:start_index]),
                    "connective_text": token_text,
                    "connective_first_token_index": first_token_index,
                    "start_index": start_index,
                    "end_index": i,
                })
    return positions


def extract_u_conn_grad(
    args,
    dataset,
    logical_connective_set,
    positions: List[int] = [-1],
    max_prompts: Optional[int] = None,
    max_length: int = 1024,
    use_logprob: bool = True,
):
    meta = {
        "model_name": args.model,
        "layer_index": str(args.layer_index),
        "token_search_window": str(getattr(args, "token_search_window", 4)),
        "prompt_per_sample": str(getattr(args, "prompt_per_sample", 1)),
        "saving_step": str(getattr(args, "saving_step", 1000)),
        "save_dir": args.save_dir,
        "dtype": str(args.dtype),
        "device": str(args.device),
        "max_prompts": str(getattr(args, "max_prompts", max_prompts)),
        "saving_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map="auto",
        token=getattr(args, "token", None),
    ).eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        token=getattr(args, "token", None),
    )

    target_layer = get_block_modules(model)[args.layer_index]
    ln_after_attn = extract_after_attention(target_layer)

    S_conn: Set[int] = build_connective_first_token_set(tokenizer)
    if not S_conn:
        raise RuntimeError("Empty connective set.")

    accum = torch.zeros([target_layer.hidden_size], dtype=torch.float32, device="cpu")
    count = 0

    pos_base = positions[:] if positions else [-1]
    z_ref: Dict[str, Optional[torch.Tensor]] = {"z": None}

    def ln_forward_hook(module, inp, out):
        h = out.detach()
        B, T, D = h.shape
        if B != 1:
            return out
        idxs = [(T + i if i < 0 else i) for i in pos_base]
        for idx in idxs:
            if not (0 <= idx < T):
                return out

        z = h[:, idxs, :].clone().to(out.dtype).requires_grad_(True)
        h_new = h.clone()
        h_new[:, idxs, :] = z
        z_ref["z"] = z
        return h_new

    handle = ln_after_attn.register_forward_hook(ln_forward_hook, with_kwargs=False)

    seen = 0
    try:
        for item in tqdm(dataset, desc="Extracting u_conn (gold, grad)"):
            if (max_prompts is not None) and (seen >= max_prompts):
                break

            prompt = item["prompt"]
            pos_list = logical_connective_position(
                text=prompt,
                tokenizer=tokenizer,
                logical_connective_set=logical_connective_set,
                token_search_window=getattr(args, "token_search_window", 4),
            )
            if not pos_list:
                continue

            pos_list = random.sample(
                pos_list,
                min(getattr(args, "prompt_per_sample", len(pos_list)), len(pos_list)),
            )

            for pos in pos_list:
                prefix = pos["prefix_text"]
                if not prefix or prefix.isspace():
                    continue

                gold_id = int(pos["connective_first_token_index"])
                if gold_id in (
                    getattr(tokenizer, "bos_token_id", -100),
                    getattr(tokenizer, "eos_token_id", -100),
                ):
                    continue

                encoded_inputs = tokenizer(
                    prefix,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                ).to(getattr(args, "device", "cuda"))

                with _autocast_ctx(args):
                    with torch.set_grad_enabled(True):
                        out = model(**encoded_inputs, use_cache=False, return_dict=True)

                z = z_ref["z"]
                if z is None:
                    del out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                logits = out.logits[:, -1, :]
                f = (F.log_softmax(logits, dim=-1)[0, gold_id] if use_logprob else logits[0, gold_id])

                g = torch.autograd.grad(f, z, retain_graph=False, create_graph=False)[0]

                g = g[0]
                g = g.mean(dim=0)

                g = g.detach().to(torch.float32).cpu()
                g = F.normalize(g, dim=-1)

                accum += g
                count += 1

                z_ref["z"] = None
                seen += 1

                del f, logits, out, g
                if torch.cuda.is_available() and (seen % max(32, getattr(args, "saving_step", 1000)) == 0):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if seen % getattr(args, "saving_step", 1000) == 0:
                    v = accum / float(count)
                    u = F.normalize(v, dim=-1)
                    fname = f"{args.model.replace('/', '_')}_u_conn_{seen}.safetensors"
                    save_path = os.path.join(args.save_dir, fname)
                    tensors = {"u_conn": u}
                    meta["total_count"] = str(count)
                    meta["saving_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_file(tensors, save_path, metadata=meta)
                    print(f"[saved] u_conn to: {save_path}")

    finally:
        handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if count == 0:
        raise RuntimeError("No samples accumulated for u_conn (gold, grad).")

    v = accum / float(count)
    u_conn = F.normalize(v, dim=-1)

    fname = f"{args.model.replace('/', '_')}_u_conn.safetensors"
    save_path = os.path.join(args.save_dir, fname)
    tensors = {"u_conn": u_conn}
    meta["total_count"] = str(count)
    meta["saving_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_file(tensors, save_path, metadata=meta)
    print(f"[saved] u_conn to: {save_path}")

    return u_conn
