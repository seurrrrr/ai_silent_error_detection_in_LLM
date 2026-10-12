"""Supervised LoRA task adaptation.
Packaging adaptation: portable file references; numerical operations retained.
"""
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict
import inspect
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def choose_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return (torch.bfloat16, True, False)
        else:
            return (torch.float16, False, True)
    return (torch.float32, False, False)

class QwenTrajectorySFTDataset(Dataset):
    """
    只对 assistant 部分计算 loss：
    - system + user = prompt
    - assistant = target
    """

    def __init__(self, rows, tokenizer, max_length=2048):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = row['messages']
        prompt_messages = messages[:-1]
        assistant_message = messages[-1]['content']
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_messages = prompt_messages + [{'role': 'assistant', 'content': assistant_message}]
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_tok = self.tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        full_tok = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids = full_tok['input_ids']
        attention_mask = full_tok['attention_mask']
        prompt_len = len(prompt_tok['input_ids'])
        prompt_len = min(prompt_len, len(input_ids))
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class DataCollatorForCausalLMPad:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max((len(x['input_ids']) for x in features))
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for f in features:
            input_ids = f['input_ids']
            attention_mask = f['attention_mask']
            labels = f['labels']
            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)
        return {'input_ids': torch.tensor(batch_input_ids, dtype=torch.long), 'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long), 'labels': torch.tensor(batch_labels, dtype=torch.long)}

def print_trainable_params(model):
    trainable = 0
    total = 0
    for (_, p) in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    ratio = 100 * trainable / total if total > 0 else 0
    print(f'[INFO] trainable params: {trainable:,}')
    print(f'[INFO] total params: {total:,}')
    print(f'[INFO] trainable ratio: {ratio:.4f}%')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--val_jsonl', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    args = parser.parse_args()
    train_jsonl = Path(args.train_jsonl)
    val_jsonl = Path(args.val_jsonl) if args.val_jsonl else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('[INFO] Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    (compute_dtype, use_bf16, use_fp16) = choose_dtype()
    print(f'[INFO] compute_dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}')
    print('[INFO] Loading model in 4-bit...')
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, quantization_config=quant_config, device_map='auto')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)
    print('[INFO] Loading datasets...')
    train_rows = load_jsonl(train_jsonl)
    val_rows = load_jsonl(val_jsonl) if val_jsonl and val_jsonl.exists() else []
    train_dataset = QwenTrajectorySFTDataset(train_rows, tokenizer=tokenizer, max_length=args.max_length)
    eval_dataset = None
    if len(val_rows) > 0:
        eval_dataset = QwenTrajectorySFTDataset(val_rows, tokenizer=tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForCausalLMPad(tokenizer)
    print(f'[INFO] train samples={len(train_dataset)}')
    print(f'[INFO] eval samples={(len(eval_dataset) if eval_dataset is not None else 0)}')
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta_kwargs = dict(output_dir=str(output_dir), num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, weight_decay=args.weight_decay, logging_steps=args.logging_steps, save_steps=args.save_steps, eval_steps=args.eval_steps, save_total_limit=2, bf16=use_bf16, fp16=use_fp16, dataloader_num_workers=0, remove_unused_columns=False, report_to='none', seed=args.seed, optim='paged_adamw_8bit', max_grad_norm=1.0, save_strategy='steps', label_names=['labels'])
    if eval_dataset is not None:
        if 'evaluation_strategy' in supported_args:
            ta_kwargs['evaluation_strategy'] = 'steps'
        elif 'eval_strategy' in supported_args:
            ta_kwargs['eval_strategy'] = 'steps'
    elif 'evaluation_strategy' in supported_args:
        ta_kwargs['evaluation_strategy'] = 'no'
    elif 'eval_strategy' in supported_args:
        ta_kwargs['eval_strategy'] = 'no'
    if 'load_best_model_at_end' in supported_args:
        ta_kwargs['load_best_model_at_end'] = False
    ta_kwargs = {k: v for (k, v) in ta_kwargs.items() if k in supported_args}
    training_args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    print('[INFO] Start training...')
    trainer.train()
    print('[INFO] Saving adapter + tokenizer...')
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    train_meta = {'model_name_or_path': args.model_name_or_path, 'train_jsonl': str(train_jsonl), 'val_jsonl': str(val_jsonl) if val_jsonl else '', 'max_length': args.max_length, 'num_train_epochs': args.num_train_epochs, 'per_device_train_batch_size': args.per_device_train_batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'learning_rate': args.learning_rate, 'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout, 'train_samples': len(train_rows), 'val_samples': len(val_rows)}
    with open(output_dir / 'train_meta.json', 'w', encoding='utf-8') as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)
    print(f'[DONE] Training finished. Adapter saved to: {output_dir}')
if __name__ == '__main__':
    main()
