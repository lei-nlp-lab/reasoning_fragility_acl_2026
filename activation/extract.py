# run.py
import argparse
import os
from transformers import AutoTokenizer
from utils import get_config
from datas import get_dataset
from extract_u_conn_act import extract_u_conn_grad 
from token_utils import get_logical_connective_set

def parse_args():
    p = argparse.ArgumentParser(description="Extract u_conn activation (attn_out, minimal)")
    p.add_argument("-m", "--model", type=str, default="google/gemma-3-4b-it")
    p.add_argument("-t", "--token", type=str)
    p.add_argument("--max-prompts", type=int, default=1000000)
    p.add_argument("--saving-step", type=int, default=20000, help="0 = disable interim save")
    p.add_argument("--save-dir", type=str, default="./store/u_conn_act")
    p.add_argument("--layer-index", type=int, default=-2)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--max-length", type=int, default=50000)
    p.add_argument("--dataset-name", type=str, default="open_thoughts")
    p.add_argument("--token-search-window", type=int, default=4)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--prompt-per-sample", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=args.token)

    dataset_config = get_config("open_thoughts")
    dataset_loader = get_dataset(args.dataset_name)
    dataset = dataset_loader(config=dataset_config, tokenizer=tokenizer)
    dataset = dataset.load()

    logical_connective_set = get_logical_connective_set()

    os.makedirs(args.save_dir, exist_ok=True)
    extract_u_conn_grad(
        args=args, 
        dataset=dataset, 
        logical_connective_set=logical_connective_set,
    )
