import warnings
import sys, os

# Ignore all warnings
warnings.filterwarnings("ignore")


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


blockPrint()

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer and model weights for calculating toxisity score of the text
tokenizer = RobertaTokenizer.from_pretrained(
    "SkolkovoInstitute/roberta_toxicity_classifier"
)
model = RobertaForSequenceClassification.from_pretrained(
    "SkolkovoInstitute/roberta_toxicity_classifier", device_map='auto'
)

def get_toxisity_score(model_output):
    sigmoid = nn.Sigmoid()
    scores = sigmoid(model_output.squeeze()).detach().numpy()
    result = {"neutral": scores[0], "toxic": scores[1]}
    return result


def get_classification(
    text, classification_tokenizer=tokenizer, classification_model=model
):
    tokenized_text = classification_tokenizer.encode(text, return_tensors="pt")
    predictions = classification_model(tokenized_text)
    return predictions

enablePrint()
print("Enter your text you want to detoxify:")
text = input()


resulting_texts = []
sample = text.split(" ")
for i, word in enumerate(sample):
    output = get_classification(word, tokenizer, model)
    result = get_toxisity_score(output.logits)
    toxicity_prob = result["toxic"]
    to_delete = np.random.rand() <= toxicity_prob
    if to_delete:
        sample[i] = "***"
resulting_texts.append(" ".join(sample))

print(resulting_texts[0])