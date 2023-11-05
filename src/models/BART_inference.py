# Import necessary libraries and modules
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
import numpy as np
import torch.nn as nn

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the base model name and the specific model name to be used
base_model_name = 'facebook/bart-base'
model_name = 'SkolkovoInstitute/bart-base-detox'

# Initialize a tokenizer using the base model name
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Initialize a BART model for conditional text generation using the specific model name
model = BartForConditionalGeneration.from_pretrained(model_name)

# Prompt the user to input the text they want to detoxify
print("Enter the text you want to detoxify:")
text = input()

# Tokenize the user's input text using the tokenizer and return it as PyTorch tensors
tokenized_text = tokenizer(text, return_tensors='pt')

# Generate detoxified text using the pre-trained model with a maximum of 512 new tokens
generated_tokens = model.generate(**tokenized_text, max_new_tokens=512)

# Decode the generated tokens into a readable detoxified text, skipping special tokens
print(tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=True))
