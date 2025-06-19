from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    return {"accuracy": accuracy}
