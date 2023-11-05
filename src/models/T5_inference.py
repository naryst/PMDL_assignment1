# Import necessary libraries and modules
import warnings
import sys, os

# Ignore all warnings to suppress warning messages
warnings.filterwarnings("ignore")

# Define functions to block and enable printing
# This is used to suppress and later restore standard output and error printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Suppress printing (stdout and stderr)
blockPrint()

# Import additional required libraries and modules
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Re-enable printing
enablePrint()

# Prompt the user to input the text they want to detoxify
print("Input your text you want to detoxify:")
text = input()

# Suppress printing again
blockPrint()

# Specify the fine-tuned model for text detoxification
# finetuned_model_chk = "../../models/T5_training_checkpoints/final_checkpoint/"
finetuned_model_chk = "narySt/T5_textDetoxificationV2"

# Initialize a tokenizer and model for text detoxification using the specified model checkpoint
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_chk)
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model_chk, device_map="auto")

# Tokenize the user's input text
text_tokenized = tokenizer(text, return_tensors="pt")
input = {
    "input_ids": text_tokenized["input_ids"],
    "attention_mask": text_tokenized["attention_mask"],
}

# Generate detoxified text using the pre-trained model with a maximum of 512 new tokens
detoxified_text_tokenized = model.generate(**input, max_new_tokens=512, do_sample=True)

# Decode the generated tokens into readable detoxified text, skipping special tokens
text_detoxified = tokenizer.decode(
    detoxified_text_tokenized[0], skip_special_tokens=True
)

# Re-enable printing for the final output
enablePrint()

# Print the detoxified text
print(text_detoxified)
