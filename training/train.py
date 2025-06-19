from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model.roberta_model import CustomRobertaForSequenceClassification
from utils.metrics import compute_metrics
from collections import Counter
import torch

def train_model(config, tokenized_datasets, tokenizer):

    label_counts = Counter(tokenized_datasets['train']['label'])
    total_count = sum(label_counts.values())
    class_weight = torch.tensor([total_count / label_counts[i] for i in range(config["num_labels"])]).float()


    model = CustomRobertaForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_labels"]
    )
    model.class_weights = class_weight

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        logging_dir=config["logging_dir"],
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
    )

    trainer.train()
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model") 
    return trainer
