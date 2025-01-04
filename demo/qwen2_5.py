# -*- coding: utf-8 -*-
# @Time : 2025/1/2 22:31
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : qwen2_5_demo.py
# @Project : ai_model_finetune
from unsloth import FastLanguageModel
import torch, os, wandb
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

LORA_MODEL_NAME = '/root/autodl-tmp/sft_model/lora_adapter'
FULL_MODEL_NAME = '/root/autodl-tmp/sft_model/full_model'

def setup_wandb(project_name: str, run_name: str):
    # Set up your API KEY
    try:
        api_key = '1d00ba7c37ca70562c00e3251790f4f0efaed1c6'
        wandb.login(key=api_key)
        print("Successfully logged into WandB.")
    except KeyError:
        raise EnvironmentError("WANDB_API_KEY is not set in the environment variables.")
    except Exception as e:
        print(f"Error logging into WandB: {e}")

    # Optional: Log models
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_SILENT"] = "true"

    # Initialize the WandB run
    try:
        wandb.init(project=project_name, name=run_name)
        print(f"WandB run initialized: Project - {project_name}, Run - {run_name}")
    except Exception as e:
        print(f"Error initializing WandB run: {e}")

setup_wandb(project_name = 'unsloth_practice', run_name='demo_20250102_demo')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/qwen2.5-13b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("json", data_files="../data/qiniu_alpaca.json", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    # num_train_epochs = 1, # Set this for 1 full training run.
    max_steps=60,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",  # Use this for WandB etc
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = args,
)

trainer_stats = trainer.train()

wandb.finish()

# Local saving
model.save_pretrained(LORA_MODEL_NAME)
tokenizer.save_pretrained(LORA_MODEL_NAME)

# Merge to 16bit
model.save_pretrained_merged(FULL_MODEL_NAME, tokenizer, save_method = "merged_16bit",)