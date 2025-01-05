import logging
logging.disable(logging.CRITICAL)

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


def load_arxiv_papers(search_query: str, max_results: int = 1):    
    loader = ArxivReader()
    documents = loader.load_data(search_query=search_query, max_results=max_results)
    return documents

class QA(BaseModel):
    question1: str = Field(description="First general knowledge question about the core topic and concepts")
    answer1: str = Field(description="Clear, factual answer focused on general knowledge")
    question2: str = Field(description="Second general knowledge question about the core topic and concepts") 
    answer2: str = Field(description="Clear, factual answer focused on general knowledge")
    question3: str = Field(description="Third general knowledge question about the core topic and concepts")
    answer3: str = Field(description="Clear, factual answer focused on general knowledge")

DEFAULT_SYSTEM_CONTENT = """You are an expert at creating high-quality training data for language models.
Given academic research content, extract the key concepts and convert them into general knowledge questions and answers.
Questions should:
- Be general and broadly applicable
- Focus on core concepts and ideas
- Be written as standalone questions without referencing any specific research
- Test understanding of the topic area

Answers should:
- Provide clear, factual information
- Be written as standalone knowledge
- Focus on general concepts rather than specific research
- Be useful for general learning about the topic"""

def generate_qa(document_text: str, system_content: str = DEFAULT_SYSTEM_CONTENT, feedback: str = None):
    json_schema = QA.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
        temperature=5.0,
        max_tokens=8000,
        min_tokens=1000,
        min_p=0.5,
        stop=["<|eot_id|>"]
    )
    
    messages = [
        {"role": "system", "content": system_content}
    ]
    
    if feedback:
        messages.append({"role": "user", "content": f"Based on this feedback, please generate improved questions and answers:\n\n{feedback}\n\nOriginal text: {document_text}"})
    else:
        messages.append({"role": "user", "content": f"Based on these concepts, generate 3 general knowledge questions and answers about: {document_text}"})
    
    return llm.chat(
        messages=messages,
        sampling_params=sampling_params
    )

class QAEvaluation(BaseModel):
    qa_pair1_arguments_for: list[str] = Field(description="Arguments in favor of including first QA pair")
    qa_pair1_arguments_against: list[str] = Field(description="Arguments against including first QA pair") 
    qa_pair1_include: Literal["y", "n"] = Field(description="Whether to include first QA pair")

    qa_pair2_arguments_for: list[str] = Field(description="Arguments in favor of including second QA pair")
    qa_pair2_arguments_against: list[str] = Field(description="Arguments against including second QA pair")
    qa_pair2_include: Literal["y", "n"] = Field(description="Whether to include second QA pair")

    qa_pair3_arguments_for: list[str] = Field(description="Arguments in favor of including third QA pair")
    qa_pair3_arguments_against: list[str] = Field(description="Arguments against including third QA pair")
    qa_pair3_include: Literal["y", "n"] = Field(description="Whether to include third QA pair")
    
    next_search_query: str = Field(description="Recommended next search query for finding related but diverse papers")

def evaluate_qa(document_text: str, qa_output: QA, round_num: int):
    json_schema = QAEvaluation.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        guided_decoding=guided_decoding_params,
        temperature=5.0,
        max_tokens=8000,
        min_tokens=1000,
        min_p=0.5,
        stop=["<|eot_id|>"]
    )
    
    eval_system_prompt = f"""You are an expert evaluator of training data quality for language models.
    This is debate round {round_num}. Be increasingly critical with each round.
    For each question-answer pair:
    1. List key arguments for and against including it (considering accuracy, clarity, generalizability)
    2. Make a binary decision (y/n) if it should be included in the final dataset
    
    Additionally, recommend a search query for finding the next paper to analyze. The query should:
    - Be related to but very distinct from the current paper's topic
    - Help build a diverse dataset while maintaining topical coherence
    - Be generic enough that it'll always return valid results
    - Potentially focus on an interesting direction suggested by the current paper"""
    
    eval_user_prompt = f"""Evaluate these QA pairs generated from the paper:
    
    Original Paper Text: {document_text}
    
    QA Pair 1:
    Q: {qa_output.question1}
    A: {qa_output.answer1}
    
    QA Pair 2:
    Q: {qa_output.question2}
    A: {qa_output.answer2}
    
    QA Pair 3:
    Q: {qa_output.question3}
    A: {qa_output.answer3}
    
    Provide a structured evaluation following the schema, including a recommended next search query."""

    messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": eval_user_prompt}
    ]
    
    return llm.chat(
        messages=messages,
        sampling_params=sampling_params
    )

def archive_enchanted_dialogues(knowledge_exchange_data, assessment_data, archive_path="wand_university_training_grimoire.csv"):
    """Archive validated magical dialogues and their scholarly assessments for the Wand University curriculum"""
    print("\n🌟 INITIATING ARCHIVAL PROCESS OF ENCHANTED DIALOGUES 🌟")
    
    # Transform assessment data from arcane notation to comprehensible format
    assessment_grimoire = json.loads(assessment_data)
    
    # Initialize collection of validated knowledge exchanges
    validated_exchanges = []
    
    # Evaluate each knowledge exchange through our rigorous magical standards
    for exchange_id in tqdm(range(1,4), desc="VALIDATING KNOWLEDGE EXCHANGES"):
        if assessment_grimoire[f'qa_pair{exchange_id}_include'] == 'y':
            # Synthesize supporting and challenging arguments
            supporting_thesis = assessment_grimoire.get(f'qa_pair{exchange_id}_arguments_for', [])
            counterpoints = assessment_grimoire.get(f'qa_pair{exchange_id}_arguments_against', [])
            
            print(f"\n✨ ACCEPTED KNOWLEDGE EXCHANGE {exchange_id}:")
            print(f"📝 INQUIRY: {getattr(knowledge_exchange_data, f'question{exchange_id}')}")
            print(f"💭 RESPONSE: {getattr(knowledge_exchange_data, f'answer{exchange_id}')}")
            print("✅ POSITIVE FEEDBACK:")
            for strength in supporting_thesis:
                print(f"  • {strength}")
            
            validated_exchanges.append({
                'inquiry': getattr(knowledge_exchange_data, f'question{exchange_id}'),
                'wisdom': getattr(knowledge_exchange_data, f'answer{exchange_id}'),
                'supporting_thesis': ', '.join(supporting_thesis),
                'counterpoints': ', '.join(counterpoints),
                'search_query': assessment_grimoire['next_search_query']
            })
        else:
            print(f"\n⚠️ DISCARDED KNOWLEDGE EXCHANGE {exchange_id}:")
            print(f"📝 INQUIRY: {getattr(knowledge_exchange_data, f'question{exchange_id}')}")
            print(f"💭 RESPONSE: {getattr(knowledge_exchange_data, f'answer{exchange_id}')}")
            print("❌ CRITICAL FEEDBACK:")
            for critique in assessment_grimoire.get(f'qa_pair{exchange_id}_arguments_against', []):
                print(f"  • {critique}")
    
    print("\n📚 TRANSFORMING KNOWLEDGE INTO STRUCTURED FORMAT...")
    # Transform into structured knowledge format
    knowledge_codex = pd.DataFrame(validated_exchanges)
    
    # Preserve in the grand archives
    print("💾 PRESERVING IN THE GRAND ARCHIVES...")
    if Path(archive_path).exists():
        knowledge_codex.to_csv(archive_path, mode='a', header=False, index=False, sep='|')
    else:
        knowledge_codex.to_csv(archive_path, index=False, sep='|')
    
    print(f"✨ SUCCESSFULLY ARCHIVED {len(validated_exchanges)} EXCHANGES ✨")
    return len(validated_exchanges)

def synthesize_and_evaluate_knowledge(source_manuscript, research_focus, debate_rounds=3):
    """Synthesize and evaluate magical knowledge exchanges through our proprietary arcane processes"""
    print("\n🔮 INITIATING KNOWLEDGE SYNTHESIS AND EVALUATION 🔮")
    
    # Initial generation
    print("📖 GENERATING INITIAL KNOWLEDGE EXCHANGES...")
    knowledge_response = generate_qa(source_manuscript, research_focus)
    my_response = ""
    for response in knowledge_response:
        my_response += response.outputs[0].text
    knowledge_data = QA.parse_raw(my_response)
    
    # Debate rounds
    for round_num in range(1, debate_rounds + 1):
        print(f"\n🎭 DEBATE ROUND {round_num} OF {debate_rounds}")
        
        # Evaluate current QA pairs
        print("⚖️ EVALUATING CURRENT KNOWLEDGE EXCHANGES...")
        assessment_response = evaluate_qa(source_manuscript, knowledge_data, round_num)
        my_assessment = ""
        for response in assessment_response:
            my_assessment += response.outputs[0].text
        
        if round_num < debate_rounds:
            # Generate improved QA pairs based on feedback
            print("🔄 GENERATING IMPROVED KNOWLEDGE EXCHANGES...")
            knowledge_response = generate_qa(source_manuscript, research_focus, my_assessment)
            my_response = ""
            for response in knowledge_response:
                my_response += response.outputs[0].text
            knowledge_data = QA.parse_raw(my_response)
    
    # Archive final round exchanges
    exchanges_preserved = archive_enchanted_dialogues(knowledge_data, my_assessment)
    
    # Extract next research direction from evaluation
    assessment_grimoire = json.loads(my_assessment)
    next_research_focus = assessment_grimoire['next_search_query']
    
    return exchanges_preserved, next_research_focus
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# MAIN LOOP - Iterates N=2 times through the entire knowledge acquisition and training process
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# LLM INITIALIZATION - Set up the language model with specific parameters
#-------------------------------------------------------------------------------------------
run = wandb.init(
    project="wand-university-knowledge-acquisition",
    name=f"knowledge-acquisition-iteration-0",
    config={
        "model": "mistralai/Mistral-Large-Instruct-2411",
        "temperature": 2.0,
        "min_p": 0.3,
        "tensor_parallel_size": 8
    }
)

sampling_params = SamplingParams(temperature=2.0, min_p=0.3)
llm = LLM(model="mistralai/Mistral-Large-Instruct-2411", tensor_parallel_size=8, guided_decoding_backend = "lm-format-enforcer")
current_research_focus = "language models"
preserved_exchanges_count = 0

print("\n🎓 WAND UNIVERSITY KNOWLEDGE ACQUISITION SYSTEM INITIALIZED 🎓")
print("="*80)

#-------------------------------------------------------------------------------------------
# KNOWLEDGE ACQUISITION LOOP - Search and process research papers
#-------------------------------------------------------------------------------------------
research_cycle = 0
while research_cycle < 3:
    try:
        # Search the grand archives
        for i in tqdm(range(0,2), desc="SEARCHING"):
            print(f"\n🔍 SEARCHING ARCHIVES WITH FOCUS: {current_research_focus}")
            ancient_manuscripts = load_arxiv_papers(current_research_focus, max_results=5)
            the_text = ""
            # Randomly select one manuscript from the 5 retrieved
            selected_manuscript = random.choice(list(ancient_manuscripts))
            for manuscript in tqdm([selected_manuscript], desc="PROCESSING MANUSCRIPTS"):
                the_text += manuscript.text
            print("\n" + "="*80)
            print(f"📜 ANALYZING MAGICAL MANUSCRIPT:")
            print("="*80 + "\n")
            print(the_text[:500] + "......")  
            print("\n" + "="*80 + "\n")
            
            exchanges_preserved, next_focus = synthesize_and_evaluate_knowledge(the_text, current_research_focus)
            preserved_exchanges_count += exchanges_preserved
            print(f"📊 PRESERVED {exchanges_preserved} MAGICAL EXCHANGES. TOTAL IN ARCHIVES: {preserved_exchanges_count}")
            print(f"🎯 NEXT RESEARCH FOCUS: {next_focus}\n")
            
            # Log metrics to wandb
            wandb.log({
                "research_cycle": research_cycle,
                "exchanges_preserved": exchanges_preserved,
                "total_preserved_exchanges": preserved_exchanges_count,
                "current_research_focus": current_research_focus,
                "next_research_focus": next_focus
            })
            
            current_research_focus = next_focus
            print(f"🔄 COMPLETED RESEARCH CYCLE {research_cycle + 1}. NEXT FOCUS: {current_research_focus}")
            research_cycle += 1
        
    except Exception as e:
        print(f"⚡ MAGICAL ANOMALY DETECTED: {str(e)}")
        wandb.log({"error": str(e)})
        continue

print(f"\n🏆 GRAND ARCHIVE COMPLETE! TOTAL MAGICAL EXCHANGES PRESERVED: {preserved_exchanges_count} 🏆")
wandb.finish()

#-------------------------------------------------------------------------------------------
# CLEANUP - Free GPU memory and shutdown services so that we can do lora training
#-------------------------------------------------------------------------------------------
destroy_model_parallel()
destroy_distributed_environment()
del llm.llm_engine.model_executor
del llm
with contextlib.suppress(AssertionError):
    torch.distributed.destroy_process_group()
gc.collect()
torch.cuda.empty_cache()
ray.shutdown()
print("Successfully delete the llm pipeline and free the GPU memory.")

#-------------------------------------------------------------------------------------------
# LORA TRAINING SETUP - Initialize configuration for fine-tuning
#-------------------------------------------------------------------------------------------
model_id = "mistralai/Mistral-7B-v0.1"
output_dir = "./wand_university_lora_big"
num_train_epochs = 1
per_device_train_batch_size = 1
learning_rate = 1e-5
max_seq_length = 4096
lora_r = 256
lora_alpha = 64
lora_dropout = 0.05

#-------------------------------------------------------------------------------------------
# WANDB INITIALIZATION - Set up experiment tracking for lora 
#-------------------------------------------------------------------------------------------
lora_run = wandb.init(
    project="wand-university-lora",
    name="arcane-knowledge-transfer-incremental",
    config={
        "epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha
    }
)

#-------------------------------------------------------------------------------------------
# DATA PREPARATION - Load and process training data
#-------------------------------------------------------------------------------------------
print(f"🔮 SUMMONING TRAINING DATA FROM THE ANCIENT GRIMOIRE... 📚 [Model: {model_id}]")
df = pd.read_csv('wand_university_training_grimoire.csv', sep='|')
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
    model = PeftModel.from_pretrained(model, output_dir)
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
        "text": f"Question: {inquiry}\nAnswer: {wisdom}"
    }
    for inquiry, wisdom in zip(df['inquiry'], df['wisdom'])
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
    run_name="arcane-knowledge-transfer-incremental-0",
    overwrite_output_dir=True,
    save_strategy="no"
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
print(f"🧹 CLEANING THE LORA ENCHANTMENT ARTIFACTS... ✨")
lora_path = '/home/lain/wand_university_lora_big/adapter_model.safetensors'
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
safetensors.torch.save_file(tensors, '/home/lain/wand_university_lora_big/adapter_model.safetensors')

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

#-------------------------------------------------------------------------------------------
# MODEL EVALUATION (using lm-evaluation-harness and vllm)
#-------------------------------------------------------------------------------------------

print(f"🧪 BEGINNING MODEL EVALUATION RITUAL... 📊")
try:
    cmd = [
        "python3", "/home/lain/lm-evaluation-harness/lm_eval",
        "--model", "vllm",
        "--model_args", "pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,tensor_parallel_size=8,distributed_executor_backend=ray,enable_lora=True,max_lora_rank=256,lora_local_path=/home/lain/wand_university_lora_big",
        "--batch_size", "1",
        "--tasks", "gpqa_main_generative_n_shot",
        "--num_fewshot", "3",
        "--limit", "400", 
        "--wandb_args", "project=wand_university",
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