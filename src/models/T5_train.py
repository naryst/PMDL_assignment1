# Import necessary libraries and modules
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import datasets
from transformers import Trainer, TrainingArguments
import transformers
import matplotlib.pyplot as plt
import wandb

# Set the directory where the prepared Hugging Face dataset is stored
DATA_DIR = "../../data/interim/hf_dataset/"

# Load the prepared Hugging Face dataset from the disk
dataset = datasets.load_from_disk(DATA_DIR)

# Remove features that are not useful for fine-tuning from the dataset
dataset = dataset.remove_columns(
    ["similarity", "lenght_diff", "trn_tox", "ref_tox", "__index_level_0__"]
)

# Load a pre-trained T5 model for text paraphrasing that will be fine-tuned
checkpoint_name = 'SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt'

# Initialize the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(checkpoint_name, device_map='auto')
tokenizer = T5Tokenizer.from_pretrained(checkpoint_name, legacy=False)

# Function to tokenize translation to get true labels for training
def make_labels(example):
    example['labels'] = tokenizer(example["translation"], return_tensors="pt",
                                padding='max_length', truncation=True)['input_ids']
    example["labels"][example['labels'] == tokenizer.pad_token_id] = -100
    return example

# Tokenize references to get input ids and attention mask for the model input
dataset = dataset.map(lambda x: tokenizer(x['reference'], return_tensors="pt", padding='max_length', truncation=True), num_proc=8)
dataset = dataset.map(make_labels, num_proc=8)
dataset.set_format('torch')

# Function to squeeze tensors for proper formatting
def squeeze(example):
    example['labels'] = torch.squeeze(example['labels'])
    example['input_ids'] = torch.squeeze(example['input_ids'])
    example['attention_mask'] = torch.squeeze(example['attention_mask'])
    return example

# Apply the squeeze function to the dataset
dataset = dataset.map(squeeze, num_proc=8)

# Split the dataset into training and validation sets
train_dataset = dataset['train']
val_dataset = dataset['test']

# Create a data collator for sequence-to-sequence models
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments for the fine-tuning process
training_args = TrainingArguments(
    output_dir='../models/T5_paraphraser',   # Output directory
    overwrite_output_dir=False,
    num_train_epochs=3,             # Total number of training epochs
    per_device_train_batch_size=32,  # Batch size per device during training
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,    # Batch size for evaluation
    warmup_steps=300,               # Number of warmup steps for the learning rate scheduler
    weight_decay=0,                  # Strength of weight decay
    learning_rate=3e-5,
    logging_dir='logs',           # Directory for storing logs
    logging_steps=100,
    eval_steps=1000,
    evaluation_strategy='steps',
    save_steps=5000,
    report_to='wandb',
    bf16=True
)

# Create a Trainer for fine-tuning the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model checkpoint
trainer.save_model('../models/T5_paraphraser/final_checkpoint')