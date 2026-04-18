import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed as hf_set_seed

try:
    from transformers.optimization import get_linear_schedule_with_warmup
    _USE_HF_SCHED = True
except Exception:
    _USE_HF_SCHED = False


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    hf_set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


def try_get_single_token_id(tokenizer, token_str: str) -> Optional[int]:
    def enc(s):
        return tokenizer.encode(s, add_special_tokens=False)

    ids = enc(token_str)
    if len(ids) == 1:
        return ids[0]

    if not token_str.startswith(" "):
        ids2 = enc(" " + token_str)
        if len(ids2) == 1:
            return ids2[0]

    if token_str.startswith(" "):
        ids3 = enc(token_str.lstrip(" "))
        if len(ids3) == 1:
            return ids3[0]

    return None


@dataclass
class PairItem:
    prompt: str
    chosen_token_id: int
    rejected_token_id: int


class NextTokenDPODataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        strict_one_token: bool = True,
        sample_limit: Optional[int] = None,
    ):
        df = pd.read_parquet(parquet_path)
        required = {"branch_text", "correct_token", "incorrect_token"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        items: List[PairItem] = []
        for _, row in df.iterrows():
            prompt = str(row["branch_text"])
            pos_tok_str = str(row["correct_token"])
            neg_tok_str = str(row["incorrect_token"])

            pos_id = try_get_single_token_id(tokenizer, pos_tok_str)
            neg_id = try_get_single_token_id(tokenizer, neg_tok_str)

            if strict_one_token:
                if pos_id is None or neg_id is None:
                    continue
            else:
                if pos_id is None:
                    pos_ids = tokenizer.encode(pos_tok_str, add_special_tokens=False)
                    if len(pos_ids) == 0:
                        continue
                    pos_id = pos_ids[0]
                if neg_id is None:
                    neg_ids = tokenizer.encode(neg_tok_str, add_special_tokens=False)
                    if len(neg_ids) == 0:
                        continue
                    neg_id = neg_ids[0]

            if pos_id == neg_id:
                continue

            items.append(PairItem(prompt=prompt,
                                  chosen_token_id=pos_id,
                                  rejected_token_id=neg_id))

        if sample_limit is not None and sample_limit > 0:
            items = items[:sample_limit]

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> PairItem:
        return self.items[idx]


class PromptCollator:
    def __init__(self, tokenizer, max_prompt_len: int):
        self.tok = tokenizer
        self.max_len = max_prompt_len
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[PairItem]) -> Dict[str, Any]:
        prompts = [x.prompt for x in batch]
        enc = self.tok(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        chosen_ids = torch.tensor([x.chosen_token_id for x in batch], dtype=torch.long)
        rejected_ids = torch.tensor([x.rejected_token_id for x in batch], dtype=torch.long)
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
        }


# ---------------------------
# Log-prob & DPO loss
# ---------------------------
def batch_next_logps(model, input_ids, attention_mask, target_ids) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]
    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    bidx = torch.arange(input_ids.size(0), device=input_ids.device)
    next_logits = logits[bidx, last_idx, :]        # [B, V]
    log_probs  = next_logits.log_softmax(dim=-1)   # [B, V]
    return log_probs.gather(1, target_ids.view(-1, 1)).squeeze(1)  # [B]


def dpo_loss_next_token(
    policy_logp_chosen, policy_logp_rejected,
    ref_logp_chosen, ref_logp_rejected,
    beta: float,
):
    pi_diff = policy_logp_chosen - policy_logp_rejected
    ref_diff = ref_logp_chosen - ref_logp_rejected
    advantage = pi_diff - ref_diff
    return -torch.nn.functional.logsigmoid(beta * advantage).mean()


def make_scheduler(optim, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    if _USE_HF_SCHED:
        return get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)
    else:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        return LambdaLR(optim, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./store_sampled/pairs.parquet")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--ref_model_name", type=str, default=None,)
    parser.add_argument("--save_dir", type=str, default="./dpo_next_token_out")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--strict_one_token", action="store_true", default=True)
    parser.add_argument("--allow_multi_token", action="store_true")
    parser.add_argument("--sample_limit", type=int, default=0, help="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    if args.allow_multi_token:
        args.strict_one_token = False

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = detect_dtype(args.dtype)

    # Tokenizer & Models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    ref_name = args.ref_model_name or args.model_name
    ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    safe_model_max = getattr(tokenizer, "model_max_length", 4096)
    if not isinstance(safe_model_max, int) or safe_model_max > 65536:
        safe_model_max = 4096
    eff_max_len = min(args.max_prompt_len, safe_model_max)

    # Dataset / Loader
    dataset = NextTokenDPODataset(
        parquet_path=args.dataset_path,
        tokenizer=tokenizer,
        strict_one_token=args.strict_one_token,
        sample_limit=(args.sample_limit if args.sample_limit and args.sample_limit > 0 else None),
    )

    collator = PromptCollator(tokenizer, max_prompt_len=eff_max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    optim = AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = math.ceil(len(loader) / max(1, args.grad_accum_steps)) * args.epochs
    sched = make_scheduler(optim, total_steps, args.warmup_ratio)

    scaler = torch.amp.GradScaler("cuda", enabled=(torch_dtype == torch.float16))
    autocast_dtype = (
        torch.bfloat16 if torch_dtype == torch.bfloat16
        else (torch.float16 if torch_dtype == torch.float16 else torch.float32)
    )

    print(f"Samples: {len(dataset)} | Steps: {total_steps} | Warmup ratio: {args.warmup_ratio}")
    print(f"Strict one-token: {args.strict_one_token} | Beta: {args.beta}")

    global_step = 0
    policy.train(); ref.eval()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_acc = 0.0
        count = 0

        for step, batch in tqdm(enumerate(loader, start=1)):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            chosen_ids = batch["chosen_ids"].to(device, non_blocking=True)
            rejected_ids = batch["rejected_ids"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(autocast_dtype != torch.float32), dtype=autocast_dtype):
                plc_ch = batch_next_logps(policy, input_ids, attention_mask, chosen_ids)
                plc_rj = batch_next_logps(policy, input_ids, attention_mask, rejected_ids)

                with torch.no_grad():
                    ref_ch = batch_next_logps(ref, input_ids, attention_mask, chosen_ids)
                    ref_rj = batch_next_logps(ref, input_ids, attention_mask, rejected_ids)

                loss = dpo_loss_next_token(plc_ch, plc_rj, ref_ch, ref_rj, beta=args.beta)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0:
                if args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

            with torch.no_grad():
                out = policy(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                last_idx = attention_mask.sum(dim=1) - 1
                bidx = torch.arange(input_ids.size(0), device=device)
                pred_ids = out.logits[bidx, last_idx, :].argmax(dim=-1)
                acc = (pred_ids == chosen_ids).float().mean().item()

            running_loss += loss.item() * args.grad_accum_steps
            running_acc += acc
            count += 1

            if global_step % args.log_interval == 0 and step % args.grad_accum_steps == 0:
                print(f"[E{epoch}] step {global_step}/{total_steps} "
                      f"loss {running_loss/count:.4f} | acc {running_acc/count:.3f} | lr {sched.get_last_lr()[0]:.2e}")

        epoch_loss = running_loss / max(1, count)
        epoch_acc = running_acc / max(1, count)
        print(f"==> Epoch {epoch} done | loss {epoch_loss:.4f} | acc {epoch_acc:.3f}")

        ckpt_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump({"loss": epoch_loss, "acc": epoch_acc}, f, indent=2)

        save_dir = args.save_dir + f"_epoch_{epoch}"
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Training finished. Model saved to", args.save_dir)


if __name__ == "__main__":
    main()
