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

# Import additional necessary libraries and modules
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from sentence_transformers import SentenceTransformer

# Check if a CUDA-enabled GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is available() else "cpu"

# Initialize a tokenizer and model for calculating the toxicity score of text
tokenizer = RobertaTokenizer.from_pretrained(
    "SkolkovoInstitute/roberta_toxicity_classifier"
)
model = RobertaForSequenceClassification.from_pretrained(
    "SkolkovoInstitute/roberta_toxicity_classifier", device_map='auto'
)

# Function to get the toxicity score from the model's output
def get_toxicity_score(model_output):
    sigmoid = nn.Sigmoid()
    scores = sigmoid(model_output.squeeze()).detach().numpy()
    result = {"neutral": scores[0], "toxic": scores[1]}
    return result

# Function to get the classification of text as toxic or neutral
def get_classification(
    text, classification_tokenizer=tokenizer, classification_model=model
):
    tokenized_text = classification_tokenizer.encode(text, return_tensors="pt")
    predictions = classification_model(tokenized_text)
    return predictions

# Re-enable printing
enablePrint()

# Prompt the user to enter the text they want to detoxify
print("Enter your text you want to detoxify:")
text = input()

resulting_texts = []

# Split the input text into words
sample = text.split("")

# Iterate through each word in the text
for i, word in enumerate(sample):
    # Get the toxicity classification for the word
    output = get_classification(word, tokenizer, model)
    result = get_toxicity_score(output.logits)
    toxicity_prob = result["toxic"]

    # Decide whether to delete the word based on its toxicity probability
    to_delete = np.random.rand() <= toxicity_prob
    if to_delete:
        sample[i] = "***"

# Append the resulting text with potentially censored words to the list
resulting_texts.append(" ".join(sample))

# Print the resulting detoxified text
print(resulting_texts[0])