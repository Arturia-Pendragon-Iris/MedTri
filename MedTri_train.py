#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioBART-only finetuning (super simple) for radiology-style normalization.
- Input file: XLSX/JSON/JSONL with parallel pairs
  * XLSX: first two cols are [src(non-standard), tgt(standard)] unless --src_col/--tgt_col set
  * JSON (array of objects): keys "input" and "target" by default
  * JSONL: one JSON per line with the same keys
- Model: any seq2seq BioBART/T5 checkpoint (default: GanjinZero/biobart-large)
- Instruction: provide with --instruction_template to prepend to inputs

Minimal dependencies:
  pip install -U pandas openpyxl datasets transformers accelerate evaluate sacrebleu rouge-score sentencepiece

Example:
python train_biomed_style_transfer.py \
  --data_path /path/to/pairs.json \
  --model_name_or_path GanjinZero/biobart-large \
  --instruction_template "Convert radiology findings+diagnosis into the specified structured format.\n\nInput:\n{src}\n\nOutput:" \
  --text_max_len 2048 --label_max_len 1024 \
  --output_dir out/biobart_ft --num_train_epochs 3 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --bf16
"""

import os
import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Optional metrics (kept minimal)
from evaluate import load as load_eval


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pairs(path: str, src_col: Optional[str] = None, tgt_col: Optional[str] = None) -> pd.DataFrame:
    """Read parallel data into DataFrame with columns [src, tgt]. Supports .xlsx/.json/.jsonl."""
    ext = os.path.splitext(path)[1].lower()

    df = pd.read_json(path)
    s = src_col or ("input" if "input" in df.columns else df.columns[0])
    t = tgt_col or ("target" if "target" in df.columns else df.columns[1])
    df = df.rename(columns={s: "src", t: "tgt"})

    def norm(x):
        if not isinstance(x, str):
            x = "" if pd.isna(x) else str(x)
        x = x.replace("\u200b", " ").replace("\xa0", " ")
        return "\n".join(l.strip() for l in x.splitlines()).strip()

    df["src"] = df["src"].apply(norm)
    df["tgt"] = df["tgt"].apply(norm)
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)].drop_duplicates()
    return df.reset_index(drop=True)


def build_dataset(tokenizer, df: pd.DataFrame, text_max_len: int, label_max_len: int,
                  instruction_template: Optional[str]):
    def _map(batch):
        if instruction_template:
            inputs = [instruction_template.format(src=s) for s in batch["src"]]
        else:
            inputs = batch["src"]
        model_in = tokenizer(inputs, max_length=text_max_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["tgt"], max_length=label_max_len, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    ds = Dataset.from_pandas(df)
    return ds.map(_map, batched=True, remove_columns=list(df.columns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="/data_backup/Project/Text_transfer/train_2.json")
    p.add_argument("--src_col", default="input")
    p.add_argument("--tgt_col", default="target")

    p.add_argument("--model_name_or_path", type=str, default="/home/chuy/PythonProjects/Arturia_platform/Report_normalization/out/biobart_ft/checkpoint-4000")
    p.add_argument("--instruction_template", type=str, default=
    "Convert the following radiology findings (text may include final diagnosis) into a structured summary. Format: each line anatomy: findings; diagnosis-category."
    "Only include anatomies present in the text; no speculation; keep each line to one sentence; total ≤250 words."
    "Use objective imaging terms; ban words: suggestive of, could represent, likely, recommend correlation."
    "Input:{src} Output:")

    p.add_argument("--text_max_len", type=int, default=512)
    p.add_argument("--label_max_len", type=int, default=256)
    p.add_argument("--train_split", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", type=str, default="out/biobart_ft")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)

    df = read_pairs(args.data_path, args.src_col, args.tgt_col)
    n = len(df)
    cut = int(n * args.train_split)
    train_df = df.iloc[:cut].reset_index(drop=True)
    eval_df = df.iloc[cut:].reset_index(drop=True) if cut < n else df.iloc[:0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    train_ds = build_dataset(tokenizer, train_df, args.text_max_len, args.label_max_len, args.instruction_template)
    eval_ds = build_dataset(tokenizer, eval_df, args.text_max_len, args.label_max_len,
                            args.instruction_template) if len(eval_df) else None

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Minimal metrics (BLEU + ROUGE), optional if eval set exists
    sacrebleu = load_eval("sacrebleu")
    rouge = load_eval("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        r = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        out = {"bleu": bleu["score"]}
        out.update({f"rouge_{k}": v for k, v in r.items()})
        return out

    gen_kwargs = dict(max_new_tokens=args.label_max_len, num_beams=4, no_repeat_ngram_size=3)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="no",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["none"],
        generation_num_beams=4,
        generation_max_length = args.label_max_len
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    trainer.train()
    # if eval_ds is not None and len(eval_df):
    #     metrics = trainer.evaluate(**{"max_length": args.label_max_len, "num_beams": 4})
    #     print("Eval:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)


if __name__ == "__main__":
    main()
