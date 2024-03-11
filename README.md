# Llama 2 Finetuning with Custom Dataset

## Overview
This repository provides a comprehensive guide and code snippets for finetuning the Llama 2 language model using a custom dataset. Leveraging the power of Hugging Face's Transformers library, you can adapt Llama 2 to your specific natural language processing tasks with ease.


### Step 1: Acquiring Access to Llama 2
Before initiating the finetuning process, ensure you have access to the Llama 2 model from Meta. Submit a request through Meta, using the same email address for both Meta and Hugging Face accounts.

#### Modules and Libraries

```
!pip install -q -U bitsandbytes
!pip install transformers==4.31
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q datasets
!pip install evaluate
!pip install -qqq trl==0.7.1

```

#### Import modules

```
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

```

### Step 2: Preparing Your Custom Dataset
Prepare a clean and relevant dataset for your specific task. Utilize the provided code to load your dataset using Pandas and the Hugging Face Datasets library.

```
import pandas as pd
from datasets import load_dataset, Dataset

# Load your dataset
df = pd.read_csv('path/to/your/dataset.csv')
train_dataset = Dataset.from_pandas(df)

```

#### DATASET FORMAT
```

def format_instruction(text: str, score: str):
	return f"""<s>[INST]{text.strip()} YOUR PROMPT.[/INST] {score}</s>""".strip()

def generate_instruction_dataset(data_point):
    return {
        "input_text": data_point["input_text"],
        "output_text": data_point["output_text"],
        "text": format_instruction(data_point["input_text"],data_point["output_text"])
    }
def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset)
    )
dataset =process_dataset(train_dataset)

```

### Step 3: Loading the Model with Quantization
Optimize model performance and reduce memory requirements by loading Llama 2 with quantization.
```
hf_auth = "HUGGINGFACE TOKEN"
model_id = 'meta-llama/Llama-2-7b-hf'

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    quantization_config=quant_config,
    device_map="auto")

model.config.use_cache = False
model.config.pretraining_tp = 1

```
### Step 4: Loading the Tokenizer
Load the appropriate Llama 2 tokenizer for processing and encoding textual data.
```
tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_auth,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

```
### Step 5: Define Lora and Training Arguments
Set up training parameters and define the Lora schedule to optimize the finetuning process.

```
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="OUTPUT",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    
    report_to="wandb",
    run_name="MODET_FINETUNE",
)

```
### Step 6: Training the Model
Initiate the training process, adapting the code to your dataset and task.
```

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

```

### Step 7: Save the Finetuned Model
Upon completion, save the finetuned model for future use.
```
final_model_path="/finetune"

trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
```








