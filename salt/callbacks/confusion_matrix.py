import numpy as np
from tabulate import tabulate

from salt.callbacks.callback import Callback

def confusion_matrix(pred_labels, true_labels, labels):
    pred_labels = pred_labels.reshape(-1)
    true_labels = true_labels.reshape(-1)
    columns = [list(map(lambda label: f'Pred {label}', labels))]
    for true_label in labels:
        counts = []
        preds_for_label = pred_labels[np.argwhere(true_labels == true_label)]
        for predicted_label in labels:
            counts.append((preds_for_label == predicted_label).sum())
        columns.append(counts)

    headers = list(map(lambda label: f'True {label}', labels))
    rows = np.column_stack(columns)
    return tabulate(rows, headers, 'grid')

class ConfusionMatrix(Callback):
    def __init__(self, labels, logger):
        self.logger = logger
        self.labels = labels

    def on_validation_end(self, logs, outputs, gt):
        self.logger(confusion_matrix(np.sign(outputs).abs(), gt, self.labels))
