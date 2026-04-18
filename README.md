# Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains

**ACL 2026 Findings**

[![Paper](https://img.shields.io/badge/ACL_2026-Findings-blue)](https://github.com/lei-nlp-lab/reasoning_fragility_acl_2026)

> **Seunghyun Park** (Independent Researcher) &nbsp; | &nbsp; **Yuanyuan Lei** (University of Florida)

---

## Abstract

While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logic deduction, where a single transition error can propagate through the entire reasoning chain. We identify **logical connectives** (e.g., *therefore*, *however*, *but*) as primary points of this structural fragility. Through empirical analysis, we show that these tokens function as high-entropy forking points at which models frequently struggle to determine the correct logical direction.

We propose a multi-layered framework that intervenes specifically at these logic-critical junctions:

1. **Gradient-based Logical Steering** — guides internal representations toward valid reasoning subspaces
2. **Localized Branching** — resolves ambiguity via targeted lookahead search at connective pivots
3. **Targeted Transition Preference Optimization (TTPO)** — a surgical RL objective that selectively optimizes single-token preferences at logical pivots

By concentrating intervention solely on logic-critical transitions, our framework achieves a favorable accuracy–efficiency trade-off compared to global inference-time scaling methods like beam search and self-consistency.

---

## Overview
<img width="1300" height="628" alt="logical_connective_paper_overall_figure" src="https://github.com/user-attachments/assets/b5228237-caa9-43c9-b12b-639d87b2475d" />

<p align="center">
  <em>Figure: Connective-centric methods across three stages — (a) Steering, (b) Branching, and (c) TTPO.</em>
</p>

---

## Usage
### 1. Steering Vector Extraction

Extract the gradient-based steering vector from OpenThoughts data:

```bash
python run.py \
    --model google/gemma-3-4b-it \
    --dataset-name open_thoughts \
    --layer-index -1 \
    --save-dir ./store/u_conn_act \
    --max-prompts 100000 \
    --dtype bfloat16
```

### 2. Inference with Steering

Run inference with the extracted steering vector:

```bash
python test_code.py \
    --model google/gemma-3-4b-it \
    --u-conn-path ./store/u_conn_act/u_conn.safetensors \
    --dataset-name zebra_logic \
    --alpha 0.5 \
    --layer-index 33 \
    --is-test \
    --save-dir ./result/steering/
```

### 3. Inference with Branching

Run inference with localized branching at connective pivots:

```bash
python branch.py \
    --model google/gemma-3-4b-it \
    --task zebra_logic \
    --k-top 20 \
    --n-lookahead 20 \
    --is-test \
    --save-dir ./result/branching/
```

### 4. TTPO Training

Train with Targeted Transition Preference Optimization:

```bash
python train_code.py \
    --model_name google/gemma-3-4b-it \
    --dataset_path ./store/pairs.parquet \
    --batch_size 1 \
    --epochs 3 \
    --lr 1e-6 \
    --beta 0.1 \
    --save_dir ./result/ttpo/
```

---

## Citation

```bibtex
@inproceedings{park2026where,
    title={Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains},
    author={Park, Seunghyun and Lei, Yuanyuan},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
    year={2026}
}
```

---

## License

This project is licensed under the MIT License.
