from typing import Dict, Type, List

import ast
import random
import pandas as pd

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

DATASET_LOADERS = {}


def dataset_loader(name: str):
    """Decorator for registering dataset loaders."""

    def decorator(cls: Type["BaseDataset"]):
        DATASET_LOADERS[name] = cls
        return cls

    return decorator


def get_dataset(task_name: str) -> "BaseDataset":
    """Get the dataset loader for a given task name."""
    return DATASET_LOADERS.get(task_name, None)


class BaseDataset:
    """Base class for dataset loaders."""

    def __init__(self, config: Dict, tokenizer: AutoTokenizer = None):
        self.config = config
        self.tokenizer = tokenizer

    def load(self) -> Dataset:
        raise NotImplementedError("Subclasses should implement this method.")

    def make_prompt(self, batch) -> Dict:
        prompts = []
        for problem in batch["problem"]:
            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": "Let's think step by step.\n\n" + problem},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"]
            )
            prompts.append(prompt)

        return {"prompt": prompts}

    def __getitem__(self, idx) -> Dict:
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


@dataset_loader("zebra_logic")
class ZebraLogic(BaseDataset):
    """Loader for the ZebraLogic dataset."""

    def __init__(self, config: Dict, tokenizer: AutoTokenizer = None, is_test: bool = False):
        super().__init__(config, tokenizer)
        self.column_name = "prompt"
        self.dataset_name = "zebra_logic"
        self.is_test = is_test
        self.data_path = config.get("zebra_logic", "data/zebra_logic")
        self.dataset = self.load()
        

    def load(self) -> Dataset:
        dataset = load_dataset(self.config["path"], "mc_mode")
        if "test" not in dataset:
            raise ValueError("Dataset must contain a 'train' split.")
        dataset = dataset["test"]

        column_names = ["id", "puzzle", "question", "choices", "answer"]
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in column_names])

        if self.is_test:
            dataset = dataset.select(range(2700, len(dataset)))
        else:
            dataset = dataset.select(range(2700))

        dataset = dataset.map(self.make_prompt, load_from_cache_file=False, batched=True, batch_size=8)

        return dataset

    def make_prompt(self, batch) -> Dict:
        prompts = []
        for puzzle, question, choices in zip(batch["puzzle"], batch["question"], batch["choices"]):
            content = f"# Puzzle\n\n{puzzle}\n\n# Question:\n\n{question}\n\n# Choices:\n\n{choices}"
            if "Phi-4-reasoning-plus" in self.tokenizer.name_or_path:
                content = self.config["system_instruction"] + "\n\n" + content + """\n\n Think step by step."""
            else:
                content = content + "\n\n Think step by step."
            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": content},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"]
            )
            prompts.append(prompt)
        return {"prompt": prompts}


@dataset_loader("open_thoughts")
class OpenThoughts(BaseDataset):
    """Loader for the OpenThoughts dataset."""

    def __init__(self, config: Dict, tokenizer: AutoTokenizer = None):
        super().__init__(config, tokenizer)
        self.column_name = "prompt"
        self.dataset_name = "OpenThoughts"
        self.data_path = config.get("open_thoughts", "data/open_thoughts")
        self.dataset = self.load()

    def load(self) -> Dataset:
        dataset = load_dataset(self.config["path"])
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split.")
        dataset = dataset["train"]

        # randomly select 1000 samples
        # dataset = dataset.select(range(2))
        dataset = dataset.map(self.make_prompt, load_from_cache_file=False, batched=True, batch_size=8)

        return dataset


    def text_preprocess(self, text: str) -> str:
        start_thought_text = "<|begin_of_thought|>"
        start_think_text = "<think>"
        end_thought_text = "<|end_of_thought|>"
        end_think_text = "</think>"
        start_solution_text = "<|begin_of_solution|>"
        end_solution_text = "<|end_of_solution|>"
        text = text.replace(start_thought_text, start_think_text).replace(end_thought_text, end_think_text)
        text = text.replace(start_solution_text, "").replace(end_solution_text, "")
        return text

    def make_prompt(self, batch) -> Dict:
        prompts = []
        for b in batch["conversations"]:
            messages = [
                {"role": "user", "content": b[0]["value"]},
                {"role": "assistant", "content": self.text_preprocess(b[1]["value"])},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompts.append(prompt)
        return {"prompt": prompts}

@dataset_loader("logic_glue")
class LogicGlue(BaseDataset):
    """Loader for the LogicGlue dataset."""

    def __init__(self, config: Dict, tokenizer: AutoTokenizer = None, is_test: bool = False):
        super().__init__(config, tokenizer)
        self.column_name = "prompt"
        self.dataset_name = "logic_glue"
        self.data_path = config.get("logic_glue", "data/logic_glue")
        self.dataset = self.load()

    def load(self) -> Dataset:
        tasks = {
            #"bigbench_deduction": lambda: self.big_bench_deduction("bigbench_deduction"),
            #"Rulebert-Union-Rules": lambda: self.rulebert_union_rules("Rulebert-Union-Rules"),
            "logiQA_2.0": lambda: self.logic_qa_0_2("logiQA_2.0")
        }

        datasets = []
        for task in tasks:
            datasets.extend(tasks[task]()[:1000])

        return Dataset.from_list(datasets)

    def make_user_message(self, context: str, question: str, choices: List[str], answer: str) -> Dict:
        answer_index = choices.index(answer)
        content = "Let's think step by step.\n\n" + f"Context:\n{context}\nQuestion:\n\{question}\nChoices\n"

        if "Phi-4-reasoning-plus" in self.tokenizer.name_or_path:
            content = self.config["system_instruction"] + "\n\n" + content
        else:
            content = content

        for i, choice in enumerate(choices):
            content += f"{chr(65+i)}. {choice}\n"
        content += "Please reasoning step by step and select the answer in the format '\\boxed{ANSWER}'"
        messages = [
            {"role": "system", "content": self.config["system_instruction"]},
            {"role": "user", "content": content},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"])
        return prompt, answer_index

    def big_bench_deduction(self, task_name: str) -> List[Dict]:
        dataset = load_dataset(self.config["path"], task_name)["test"]
        items = []
        for i, item in enumerate(dataset):
            user_message, answer_index = self.make_user_message(item["context"], item["question"], item["choices"], item["answer_text"])
            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": user_message},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"])
            items.append({"prompt": prompt, "id": task_name + "-" + str(i), "answer": chr(65+answer_index), "tag": task_name})
        return items

    def rulebert_union_rules(self, task_name: str) -> List[Dict]:
        dataset = load_dataset(self.config["path"], task_name)["test"]

        items = []
        for i, item in enumerate(dataset):
            content = "Let's think step by step.\n\n" + "Context:\n" + item["context"] + "\nQuestion:\n" + item["question"] + "\nOptions:\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])]) + "Please reasoning step by step and select the answer in the format '\\boxed{ANSWER}'"
            
            if "Phi-4-reasoning-plus" in self.tokenizer.name_or_path:
                content = self.config["system_instruction"] + "\n\n" + content
            else:
                content = content

            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": content},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"])
            items.append({"prompt": prompt, "id": task_name + "-" + str(i), "answer": "A" if item["answer_text"] == "True" else "B", "tag": task_name})
        return items

    def logic_qa_0_2(self, task_name: str) -> List[Dict]:
        dataset = load_dataset(self.config["path"], task_name)["test"]
 
        items = []
        for i, item in enumerate(dataset):
            content = "Let's think step by step.\n\n" + "Context:\n" + item["premise"] + "\nHypothesis:\n" + item["hypothesis"] + "\nQuestion:\n" + "What is the realtion between context and hypothesis?" + "\nOptions:\n" + "\nA. not-entailment\nB. entailment" + "Please reasoning step by step and select the answer in the format '\\boxed{ANSWER}'"
            
            if "Phi-4-reasoning-plus" in self.tokenizer.name_or_path:
                content = self.config["system_instruction"] + "\n\n" + content
            else:
                content = content

            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": content},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"])
            items.append({"prompt": prompt, "id": task_name + "-" + str(i), "answer": "A" if item["answer_text"] == "not-entailment" else "B", "tag": task_name})
        return items


@dataset_loader("pronto_qa")
class ProntoQA(BaseDataset):
    """Loader for the ProntoQA dataset."""

    def __init__(self, config: Dict, tokenizer: AutoTokenizer = None, is_test: bool = False):
        super().__init__(config, tokenizer)
        self.column_name = "prompt"
        self.dataset_name = "pronto_qa"
        self.data_path = config.get("pronto_qa", "data/pronto_qa")
        self.dataset = self.load()

    def load(self) -> Dataset:
        dataset = load_dataset(self.config["path"])
        if "validation" not in dataset:
            raise ValueError("Dataset must contain a 'validation' split.")
        dataset = dataset["validation"]
        dataset = dataset.map(self.make_prompt, load_from_cache_file=False, batched=True, batch_size=8)
        return dataset
    
    def make_prompt(self, batch) -> Dict:
        prompts = []
        for context, question, options in zip(batch["context"], batch["question"], batch["options"]):
            content = "Let's think step by step.\n\n" + f"Context:\n{context}\nQuestion:\n{question}\nOptions:\n"
            for i, option in enumerate(options):
                content += f"{option}\n"

            if "Phi-4-reasoning-plus" in self.tokenizer.name_or_path:
                content = self.config["system_instruction"] + "\n\n" + content
            else:
                content = content

            messages = [
                {"role": "system", "content": self.config["system_instruction"]},
                {"role": "user", "content": content},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.config["enable_thinking"])
            prompts.append(prompt)
        return {"prompt": prompts}
