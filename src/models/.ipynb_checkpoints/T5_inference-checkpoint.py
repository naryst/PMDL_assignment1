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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

enablePrint()
print("Input your text you want to detoxify:")
text = input()
blockPrint()

# Fine-tuned model for text detoxification
# finetuned_model_chk = "../../models/T5_training_checkpoints/final_checkpoint/"
finetuned_model_chk = "narySt/T5_textDetoxificationV2"
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_chk)
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model_chk, device_map="auto")


text_tokenized = tokenizer(text, return_tensors="pt")
input = {
    "input_ids": text_tokenized["input_ids"],
    "attention_mask": text_tokenized["attention_mask"],
}
detoxified_text_tokenized = model.generate(**input, max_new_tokens=512, do_sample=True)
text_detoxified = tokenizer.decode(
    detoxified_text_tokenized[0], skip_special_tokens=True
)
enablePrint()
print(text_detoxified)
