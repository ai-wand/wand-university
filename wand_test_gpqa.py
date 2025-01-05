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


#-------------------------------------------------------------------------------------------
# LORA TRAINING SETUP - Initialize configuration for fine-tuning
#-------------------------------------------------------------------------------------------
model_id = "mistralai/Mistral-7B-v0.1"
output_dir = "./wand_university_lora_gpqa"
num_train_epochs = 2
per_device_train_batch_size = 1
learning_rate = 1e-5
max_seq_length = 4096
lora_r = 256
lora_alpha = 64
lora_dropout = 0.05

#-------------------------------------------------------------------------------------------
# WANDB INITIALIZATION - Set up experiment tracking for lora 
#-------------------------------------------------------------------------------------------
"""
lora_run = wandb.init(
    project="wand-university-lora-gpqa",
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
print(f"🔮 SUMMONING TRAINING DATA FROM THE GPQA DATASET... 📚 [Model: {model_id}]")
from datasets import load_dataset
ds = load_dataset("Idavidrein/gpqa", "gpqa_main")
#print(ds['train'].column_names)
df = pd.DataFrame({
    'question': ds['train']['Question'],
    'answer': ds['train']['Correct Answer']
})
df = df.dropna()

#-------------------------------------------------------------------------------------------
# MODEL INITIALIZATION - Load base student model and configure tokenizer, for now mistral7b
#-------------------------------------------------------------------------------------------
print(f"⚡ INVOKING THE SACRED TOKENIZER AND MODEL... 🤖 [Sequence Length: {max_seq_length}]")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', attn_implementation="flash_attention_2", torch_dtype=torch.float16)
model.enable_input_require_grads()

for param in model.parameters():
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
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=lora_dropout,
    task_type="CAUSAL_LM",
)

#-------------------------------------------------------------------------------------------
# MODEL LOADING/CREATION - Either load existing LoRA or create new one if doesn't exist
#-------------------------------------------------------------------------------------------
if os.path.exists(output_dir):
    print(f"🔄 LOADING EXISTING LORA ENCHANTMENT FROM {output_dir}...")
    model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
else:
    print(f"✨ CREATING NEW LORA ENCHANTMENT WITH RANK {lora_r} AND ALPHA {lora_alpha}...")
    model = get_peft_model(model, config)

model.print_trainable_parameters()

#-------------------------------------------------------------------------------------------
# DATASET PREPARATION - Format data for training
#-------------------------------------------------------------------------------------------
print(f"📜 PREPARING THE SACRED TEXTS... ✨ [Dataset Size: {len(df)} entries]")
formatted_data = [
    {
        "text": f"Question: {question}\nAnswer: {answer}"
    }
    for question, answer in zip(df['question'], df['answer'])
]
dataset = Dataset.from_list(formatted_data)

#-------------------------------------------------------------------------------------------
# TRAINING CONFIGURATION - Set up training arguments
#-------------------------------------------------------------------------------------------
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    learning_rate=learning_rate,
    report_to="wandb",
    run_name="arcane-knowledge-transfer-gpqa",
    overwrite_output_dir=True,
    metric_for_best_model="loss",
    greater_is_better=False,  # This is correct - for loss metric we want lower values
    save_strategy="best",
    logging_steps=1,
    logging_first_step=True,
)

#-------------------------------------------------------------------------------------------
# TRAINER INITIALIZATION AND TRAINING
#-------------------------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset
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

# Free up GPU memory
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
#-------------------------------------------------------------------------------------------
# LORA CLEANUP AND SILLY STUFF TO MAKE IT WORK IN VLLM
#-------------------------------------------------------------------------------------------
"""
print(f"🧹 CLEANING THE LORA ENCHANTMENT ARTIFACTS... ✨")
lora_path = '/home/lain/wand_university_lora_gpqa/adapter_model.safetensors'
import safetensors.torch
tensors = safetensors.torch.load_file(lora_path)

nonlora_keys = []
for k in list(tensors.keys()):
    if "lora" not in k:
        nonlora_keys.append(k)

print(f"🔍 FOUND {len(nonlora_keys)} NON-LORA KEYS TO REMOVE...")
for k in nonlora_keys:
    del tensors[k]

print(f"💾 SAVING CLEANED LORA ENCHANTMENT... 📦")
safetensors.torch.save_file(tensors, '/home/lain/wand_university_lora_gpqa/adapter_model.safetensors')


# Cleanup objects
del tensors
del nonlora_keys
destroy_model_parallel()
destroy_distributed_environment()
with contextlib.suppress(AssertionError):
    torch.distributed.destroy_process_group()
gc.collect()
torch.cuda.empty_cache()
ray.shutdown()
print("Successfully delete the llm pipeline and free the GPU memory.")
"""

#-------------------------------------------------------------------------------------------
# MODEL EVALUATION (using lm-evaluation-harness and vllm)
#-------------------------------------------------------------------------------------------

print(f"🧪 BEGINNING MODEL EVALUATION RITUAL... 📊")
try:
    """
    cmd = [
        "python3", "/home/lain/lm-evaluation-harness/lm_eval",
        "--model", "vllm",
        "--model_args", "pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,tensor_parallel_size=8,distributed_executor_backend=ray,enable_lora=True,max_lora_rank=256,lora_local_path=/home/lain/wand_university_lora_gpqa",
        "--batch_size", "1",
        "--tasks", "gpqa_main_generative_n_shot",
        "--num_fewshot", "3",
        "--wandb_args", "project=wand_university-gpqa",
        "--log_samples",
        "--output_path", "/home/lain/lm-eval-output/",
        "--gen_kwargs", "min_p=0.1,temperature=1.0,do_sample=True",
        "--device", "cuda"
    ]
    """
    cmd = [
        "python3", "/home/lain/lm-evaluation-harness/lm_eval",
        "--model", "hf",
        "--model_args", "pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,parallelize=True,peft=/home/lain/wand_university_lora_gpqa,attn_implementation=flash_attention_2",
        "--batch_size", "64",
        "--tasks", "gpqa_main_generative_n_shot",
        "--num_fewshot", "0",
        "--wandb_args", "project=wand_university-gpqa",
        "--log_samples",
        "--output_path", "/home/lain/lm-eval-output/",
        "--gen_kwargs", "min_p=0.1,temperature=1.0,do_sample=True",
        "--device", "cuda"
    ]
    process = subprocess.run(cmd, check=True)
    print(f"🧪 MODEL EVALUATION RITUAL COMPLETE! 📊")
finally:
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")