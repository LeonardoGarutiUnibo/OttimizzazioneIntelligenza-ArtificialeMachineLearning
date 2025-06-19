import json
import numpy as np
from datasets import load_dataset
import torch
import random
from utils.plotting import plot_confusion_matrix
from data.data_loader import load_and_tokenize_dataset
from training.train import train_model
from training.evaluate import evaluate_model
from transformers import RobertaTokenizer

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


with open("config.json") as f:
    config = json.load(f)


tokenizer = RobertaTokenizer.from_pretrained(config["model_name"])

tokenized_datasets, original_dataset = load_and_tokenize_dataset(
    config["model_name"], config["max_length"]
)

trainer = train_model(config, tokenized_datasets, tokenizer)

evaluate_model(trainer, tokenized_datasets, original_dataset['train'].features['label'].names)

from utils.plotting import plot_confusion_matrix

predictions = trainer.predict(tokenized_datasets['test'])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

dataset = load_dataset("emotion")

plot_confusion_matrix(y_true, y_pred, dataset['train'].features['label'].names)
