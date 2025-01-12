import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import gc
import contextlib
import ray
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, timedelta
from lmformatenforcer import JsonSchemaParser
from openai import OpenAI
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import pprint
import random
from fundus import PublisherCollection, Crawler
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from llama_index.readers.papers import ArxivReader
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import wandb
import os
from enum import Enum
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import subprocess
from lm_eval import evaluator, models
from lm_eval.models.huggingface import HFLM


#-------------------------------------------------------------------------------------------
# LORA TRAINING SETUP - Initialize configuration for fine-tuning
#-------------------------------------------------------------------------------------------
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
output_dir = "./wand_university_lora_gsm8k-mistral-7b-lora-instruct-v0.3"
num_train_epochs = 100000
per_device_train_batch_size = 8
learning_rate = 1e-6
max_seq_length = 4096
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

#-------------------------------------------------------------------------------------------
# WANDB INITIALIZATION - Set up experiment tracking for lora 
#-------------------------------------------------------------------------------------------
"""
lora_run = wandb.init(
    project="wand-university-lora-gsm8k",
    name="arcane-knowledge-transfer-incremental",
    config={
        "epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha
    }
)
"""
#-------------------------------------------------------------------------------------------
# DATA PREPARATION - Load and process training data
#-------------------------------------------------------------------------------------------
print(f"🔮 SUMMONING TRAINING DATA FROM THE GSM8K DATASET... 📚 [Model: {model_id}]")
from datasets import load_dataset
ds = load_dataset("gsm8k", "main")

def format_example(example):
    return {
        'text': f"Question: {example['question']}\nAnswer:",
        'target': example['answer']
    }

ds = ds.map(format_example)

#-------------------------------------------------------------------------------------------
# MODEL INITIALIZATION - Load base student model and configure tokenizer, for now mistral7b
#-------------------------------------------------------------------------------------------
print(f"⚡ INVOKING THE SACRED TOKENIZER AND MODEL... 🤖 [Sequence Length: {max_seq_length}]")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', attn_implementation="flash_attention_2", torch_dtype="auto", trust_remote_code=True)
base_model.enable_input_require_grads()

for param in base_model.parameters():
    param.requires_grad = True

#-------------------------------------------------------------------------------------------
# LORA CONFIGURATION - Set up LoRA parameters and architecture
#-------------------------------------------------------------------------------------------
config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
    # Apply to all layers by not specifying layers_to_transform
)

#-------------------------------------------------------------------------------------------
# MODEL LOADING/CREATION - Either load existing LoRA or create new one if doesn't exist
#-------------------------------------------------------------------------------------------
if os.path.exists(output_dir):
    print(f"🔄 LOADING EXISTING LORA ENCHANTMENT FROM {output_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, output_dir, is_trainable=True)
else:
    print(f"✨ CREATING NEW LORA ENCHANTMENT WITH RANK {lora_r} AND ALPHA {lora_alpha}...")
    peft_model = get_peft_model(base_model, config)

peft_model.print_trainable_parameters()

#-------------------------------------------------------------------------------------------
# EVALUATION MODEL SETUP
#-------------------------------------------------------------------------------------------
eval_model = HFLM(
    pretrained=base_model,
    peft=peft_model,
    device="auto",
    dtype="auto",
    batch_size=64,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

def compute_metrics(eval_pred):
    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=["gsm8k"],
        num_fewshot=5,
        log_samples=True
    )
    #print(results)
    
    # Extract all GSM8K related scores
    gsm8k_results = results['results']['gsm8k']
    metrics = {
        "gsm8k_strict_match": gsm8k_results['exact_match,strict-match'],
        "gsm8k_flexible_extract": gsm8k_results['exact_match,flexible-extract']
    }

    print(f"Evaluation Metrics:")
    print(f"GSM8K Strict Match Score: {metrics['gsm8k_strict_match']:.2f}")
    print(f"GSM8K Flexible Extract Score: {metrics['gsm8k_flexible_extract']:.2f}")
    
    return metrics

#-------------------------------------------------------------------------------------------
# TRAINING CONFIGURATION - Set up training arguments
#-------------------------------------------------------------------------------------------
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
    report_to="wandb",
    run_name="arcane-knowledge-transfer-gsm8k-lora-mistral-7b-instruct-v0.3",
    overwrite_output_dir=True,
    metric_for_best_model="eval_gsm8k_flexible_extract",
    greater_is_better=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=1,
    save_total_limit=5,
    logging_first_step=True,
    load_best_model_at_end=True,
    bf16=True,
)

#-------------------------------------------------------------------------------------------
# TRAINER INITIALIZATION AND TRAINING
#-------------------------------------------------------------------------------------------
trainer = SFTTrainer(
    model=peft_model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    compute_metrics=compute_metrics,
    peft_config=config
)

print(f"🌟 COMMENCING THE SACRED LORA TRAINING RITUAL... 🧙‍♂️ [Epochs: {num_train_epochs}, Learning Rate: {learning_rate}]")
trainer.train()

#-------------------------------------------------------------------------------------------
# MODEL SAVING AND CLEANUP
#-------------------------------------------------------------------------------------------
print(f"💾 PRESERVING THE ENHANCED MODEL IN THE ARCANE ARCHIVES... 📦 [Output Directory: {output_dir}]")
trainer.model.save_pretrained(output_dir)

print(f"📊 CONCLUDING THE WANDB LOGGING RITUAL... 🔮")
wandb.finish()

print(f"🎉 THE SACRED LORA TRAINING RITUAL IS COMPLETE! 🎊 [Total Training Steps: {trainer.state.global_step}]")
