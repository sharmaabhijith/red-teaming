"""Fine-tuning Instructor LLM on a blended Evol-Instruct + OpenCodeInstruct corpus.

Prerequisites (one‑time):
    pip install -U "transformers>=4.40.0" "datasets>=2.18.0" peft accelerate bitsandbytes nltk tap

Typical usage:
    1) ----- Custom Blended dataset (60% Evol, 40% OCI) -----
    python instruct_trainer.py --datatype blended --evol_weight 0.6 --oci_weight 0.4 \
        --epochs 12 --batch_size 8
    2) ----- EvolInstruct only 
    python instruct_trainer.py --datatype evol --epochs 12 --batch_size 8
    3) ----- OpenCodeInstruct only -----
    python instruct_trainer.py --datatype oci --epochs 12 --batch_size 8
    
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import nltk
import torch
from datasets import IterableDataset, interleave_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from tap import Tap                      # Typed Argument Parser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

nltk.download("punkt", quiet=True)

# -----------------------------------------------------------------------------
# 1.  Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class ExpanderConfig:
    """Centralised hyper‑parameter & prompt store."""

    # Model / training
    base_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    out_dir: str = "deepseek_6.7b_instr_expander"
    num_epochs: int = 1
    batch_size: int = 4
    grad_acc_steps: int = 8
    lr: float = 2e-4

    # Dataset mixing
    evol_weight: float = 0.8
    oci_weight: float = 0.2
    dataset_type: str = "blended"          # {"blended", "evol", "oci"}
    seed: int = 42

    # LoRA hyper‑params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Prompt templates
    sys_tag: str = "<|system|>You are an expert programming tutor."
    usr_tag: str = "<|user|>{problem}"
    ass_tag: str = "<|assistant|>{constraints}"

    def prompt(self, problem: str) -> str:  # helper used during tokenisation
        return (
            f"{self.sys_tag}\n"
            f"{self.usr_tag.format(problem=problem)}\n"
            f"{self.ass_tag.format(constraints='')}"
        )


# -----------------------------------------------------------------------------
# 2.  Main trainer class
# -----------------------------------------------------------------------------


class InstructionTrainer:
    """Build dataset, attach LoRA adapters, and train."""

    def __init__(self, cfg: ExpanderConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._init_lora_model()

    # ------------------------------------------------------------------
    # Dataset utils
    # ------------------------------------------------------------------

    @staticmethod
    def _split_evol(row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Take first sentence as *input*, rest as *target*."""
        sents = nltk.sent_tokenize(row["instruction"])
        if len(sents) < 2:
            return None
        return {"input": sents[0].strip(), "target": " ".join(sents[1:]).strip()}

    @staticmethod
    def _clean_oci(row: Dict[str, str]) -> Optional[Dict[str, str]]:
        if row.get("generation_algorithm") != "cot":
            return None
        prose = row["output"].split("```", 1)[0]
        prose = re.sub(r"(?i)^solution\s*[:\-]*\s*", "", prose).strip()
        if len(prose.split()) < 8:
            return None
        return {"input": row["input"].strip(), "target": prose}

    def _encode(self, sample: Dict[str, str]):
        prompt = self.cfg.prompt(sample["input"])
        features = self.tokenizer(prompt, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                sample["target"] + self.tokenizer.eos_token, return_tensors="pt"
            )
        features["labels"] = labels["input_ids"]
        return features

    @staticmethod
    def _collate(batch):
        return {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0]}

    # -------------------------------------------------------------------------
    def build_dataset(self) -> IterableDataset:
        """Return HF IterableDataset according to cfg.dataset_type."""

        evol = (
            load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
            .map(self._split_evol, remove_columns=["instruction", "output"])
            .filter(lambda x: x is not None)
        )

        oci = (
            load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
            .map(self._clean_oci)
            .filter(lambda x: x is not None)
        )

        if self.cfg.dataset_type == "evol":
            dataset = evol
        elif self.cfg.dataset_type == "oci":
            dataset = oci
        else:  # blended (default)
            dataset = interleave_datasets(
                [evol, oci],
                probabilities=[self.cfg.evol_weight, self.cfg.oci_weight],
                seed=self.cfg.seed,
                stopping_strategy="all_exhausted",
            )

        return dataset.map(self._encode, remove_columns=dataset.column_names, batched=False)

    # ------------------------------------------------------------------
    #  Model / training helpers
    # ------------------------------------------------------------------

    def _init_lora_model(self):
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        return get_peft_model(base, lora_cfg)

    def train(self):
        dataset = self.build_dataset()
        args = TrainingArguments(
            output_dir=self.cfg.out_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_acc_steps,
            learning_rate=self.cfg.lr,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=50,
            save_steps=1000,
            report_to="none",
        )
        Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=self._collate,
        ).train()
        self.model.save_pretrained(f"{self.cfg.out_dir}/final")
        self.tokenizer.save_pretrained(f"{self.cfg.out_dir}/final")

    # -------------------------------------------------------------------------
    def generate(self, problem: str, **gen_kw):
        """Quick helper to test the fine‑tuned adapter."""
        messages = [{"role": "user", "content": problem}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        gen = self.model.generate(inputs, **gen_kw)
        return self.tokenizer.decode(gen[0][inputs.shape[-1] :], skip_special_tokens=True)


# -----------------------------------------------------------------------------
# 3.  CLI entry‑point (Tap)
# -----------------------------------------------------------------------------


class ExpanderArgs(Tap):
    """Tap‑based CLI. All types are auto‑validated."""

    epochs: int = 1                          # training epochs
    batch_size: int = 4                      # per‑device batch size
    evol_weight: float = 0.8                 # probability for Evol‑Instruct
    oci_weight: float = 0.2                  # probability for OpenCodeInstruct
    datatype: str = "blended"                # {"blended", "evol", "oci"}
    out_dir: str = "deepseek_6.7b_instr_expander"

    def configure(self):
        self.add_argument("--datatype", choices=["blended", "evol", "oci"], help="Which dataset split to use")


def parse_args() -> ExpanderConfig:
    args = ExpanderArgs().parse_args()
    return ExpanderConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        evol_weight=args.evol_weight,
        oci_weight=args.oci_weight,
        dataset_type=args.datatype,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    trainer = InstructionTrainer(cfg)
    trainer.train()

    # Sanity check
    prompt = "Write a Python function that determines whether two strings are anagrams."
    print("\nSample generation:\n", trainer.generate(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9))
