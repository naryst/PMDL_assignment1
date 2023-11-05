import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import datasets
from transformers import Trainer, TrainingArguments
import transformers
import matplotlib.pyplot as plt
import wandb


# load prepared hf dataset from the disk
DATA_DIR = "../data/interim/hf_dataset/"
dataset = datasets.load_from_disk(DATA_DIR)

# remove features, which is useless for fine-tuning from the dataset
dataset = dataset.remove_columns(
    ["similarity", "lenght_diff", "trn_tox", "ref_tox", "__index_level_0__"]
)

# load pretrained model (T5 for text perephrasing), which I will finetune
checkpoint_name = 'SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt'

model = T5ForConditionalGeneration.from_pretrained(checkpoint_name, device_map='auto')
tokenizer = T5Tokenizer.from_pretrained(checkpoint_name, legacy=False)


# tokenize translation to get true labels for training
def make_labels(example):
    example['labels'] = tokenizer(example["translation"], return_tensors="pt",
                                padding='max_length', truncation=True)['input_ids']
    example["labels"][example['labels'] == tokenizer.pad_token_id] = -100
    return example

# tokenize references to get input ids and attention mask for the model input
dataset = dataset.map(lambda x: tokenizer(x['reference'], return_tensors="pt", padding='max_length', truncation=True), num_proc=8)
dataset = dataset.map(make_labels, num_proc=8)
dataset.set_format('torch')

def squeeze(example):
    example['labels'] = torch.squeeze(example['labels'])
    example['input_ids'] = torch.squeeze(example['input_ids'])
    example['attention_mask'] = torch.squeeze(example['attention_mask'])
    return example

dataset = dataset.map(squeeze, num_proc=8)

train_dataset = dataset['train']
val_dataset = dataset['test']


data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = TrainingArguments(
    output_dir='../models/T5_paraphraser',   # output directory
    overwrite_output_dir=False,
    num_train_epochs=3,             # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,    # batch size for evaluation
    warmup_steps=300,               # number of warmup steps for learning rate scheduler
    weight_decay=0,                  # strength of weight decay
    learning_rate=3e-5,
    logging_dir='logs',           # directory for storing logs
    logging_steps=100,
    eval_steps=1000,
    evaluation_strategy='steps',
    save_steps=5000,
    report_to='wandb',
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model('../models/T5_paraphraser/final_checkpoint')
