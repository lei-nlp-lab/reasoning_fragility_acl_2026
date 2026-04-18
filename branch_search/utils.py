import random

import numpy as np
import torch
import wandb
import yaml
from transformers import is_torch_available


def wandb_setting(config: dict):
    wandb.init(
        project=config["wandb_project"],
        name=config["save_name"],
        config=config,
    )


def wandb_finish() -> None:
    wandb.finish()


def set_seed(seed: int = 9728):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config(name: str) -> dict:
    with open(f"configs/{name}.yaml", "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)
    return args
