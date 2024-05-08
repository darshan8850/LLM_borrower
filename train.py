import numpy as np
import pandas as pd
import inspect 

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model,TaskType
from trl import DataCollatorForCompletionOnlyLM,SFTTrainer
from torch.utils.data import DataLoader

from huggingface_hub.hf_api import HfFolder
from datasets import Dataset

import gdown

hf_api_key = "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"
HfFolder.save_token(hf_api_key)

model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
pad_token="<pad>"


def download_file_from_google_drive(url, output):
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)
    
google_drive_url = "https://drive.google.com/file/d/your_file_id_here/view?usp=sharing"
output_file_path = "data.csv"


download_file_from_google_drive(google_drive_url, output_file_path)

print("File downloaded successfully!")

df2 = pd.read_csv("./data.csv")

train, test = train_test_split(df2, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

print("Training set shape:", train.shape)
print("Validation set shape:", val.shape)
print("Test set shape:", test.shape)

dataset={
    "train": Dataset.from_pandas(train),
    "val": Dataset.from_pandas(val),
    "test": Dataset.from_pandas(test)
}

tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": pad_token})
tokenizer.padding_side = "right"

model=AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

def format_example(example):
    return inspect.cleandoc(f"""
        #Income Range
        {example["Income Range"]}
        
        # Employment Status:
        {example["Employment Status"]}
        
        # Amount Owed:
        {example["Amount Owed"]}
        
        # Type of Debt:
        {example["Type of Debt"]}
        
        # Delinquency Status:
        {example["Delinquency Status"]}
        
        #Transcripts:
        {example["Transcripts"]}
        
        # Payment History:
        {example["Payment History"]}
        
        # Financial Hardship Indicator:
        {example["Financial Hardship Indicator"]}
        
        # Communication Preference:
        {example["Communication Preference"]}
        
        # Formality Level:
        {example["Formality Level"]}
        
        #Sentiment_Labels:
        {example["Sentiment Labels"]}
    """)

model.pad_token_id=tokenizer.pad_token_id
model.config.pad_token_id=tokenizer.pad_token_id

lora_config= LoraConfig(r=128,lora_alpha=128,target_modules=["self_atth.q_proj",
                                                             "self_attn.k_proj", 
                                                             "self_attn.v_proj", 
                                                             "self_attn.o_proj",
                                                             "mlp.gate_proj",
                                                             "mlp.up_proj",
                                                             "mlp.down_proj"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM,)
model=get_peft_model(model, lora_config)
model.print_trainable_parameters()

response_template_with_context = "\n# Sentiment_Labels:"
response_template_ids= tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
examples = [format_example(dataset["train"][0])] 
encodings = [tokenizer(e) for e in examples]
dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)


batch = next(iter(dataloader))

SEED = 42 

training_arguments = TrainingArguments(       
    output_dir="experiments",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optimizer_type="adamw",  # Use "adamw" instead of "optim"
    evaluation_strategy="steps",
    eval_steps=500,  # Assuming you want to evaluate every 500 steps
    
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # Enable FP16 training
    save_strategy="epoch",
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",  # Use "linear" for a linear learning rate scheduler
    report_to="tensorboard",
    save_total_limit=5,  # Maximum number of checkpoints to save
    seed=SEED
)


def format_prompts(example):
    
    output_texts = []
    for i in range(len(example['title'])):
      text= inspect.cleandoc(f"""

        #Income Range
        {example["Income Range"][i]}
        
        # Employment Status:
        {example["Employment Status"][i]}
        
        # Amount Owed:
        {example["Amount Owed"][i]}
        
        # Type of Debt:
        {example["Type of Debt"][i]}
        
        # Delinquency Status:
        {example["Delinquency Status"][i]}
        
        #Transcripts:
        {example["Transcripts"][i]}
        
        # Payment History:
        {example["Payment History"][i]}
        
        # Financial Hardship Indicator:
        {example["Financial Hardship Indicator"][i]}
        
        # Communication Preference:
        {example["Communication Preference"][i]}
        
        # Formality Level:
        {example["Formality Level"][i]}
    
        #Sentiment_Labels:
        {example["Sentiment Labels"][i]}
    """)
      
      output_texts.append(text)
    return output_texts

trainer= SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        max_seq_length=1024,
        formatting_func=format_prompts,
        data_collator=collator
)

trainer.train()
trainer.model.save_pretrained("borrower_llm")
tokenizer.save_pretrained("borrower_llm")
