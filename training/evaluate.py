from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(trainer, tokenized_datasets, label_names):
    predictions = trainer.predict(tokenized_datasets['test'])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(classification_report(y_true, y_pred, target_names=label_names))
