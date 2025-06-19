from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np

model_path = "./final_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()
    
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
print("Testo:", text)
print("Emozione predetta:", labels[prediction])