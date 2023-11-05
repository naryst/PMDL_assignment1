import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
import numpy as np
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model_name = 'facebook/bart-base'
model_name = 'SkolkovoInstitute/bart-base-detox'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

print("Enter the text you want to detoxify:")
text = input()

tokenized_text = tokenizer(text, return_tensors='pt')
generated_tokens = model.generate(**tokenized_text, max_new_tokens=512)
print(tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=True))