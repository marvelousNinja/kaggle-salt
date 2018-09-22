import matplotlib.pyplot as plt
import numpy as np

from salt.callbacks.callback import Callback
from salt.utils import from_numpy
from salt.utils import to_numpy

def visualize_predictions(image_logger, max_samples, metric_fn, logits, gt):
    num_samples = min(len(gt), max_samples)
    gt = gt[:num_samples]
    logits = logits[:num_samples].squeeze()
    probs = 1 / (1 + np.exp(-logits))

    metrics = to_numpy(metric_fn(from_numpy(logits), from_numpy(gt), average=False))
    order = np.argsort(metrics)
    metrics = metrics[order]
    probs = probs[order]
    gt = gt[order]

    samples_per_row = 16
    num_rows = int(np.ceil(num_samples / samples_per_row)) * 2
    plt.figure(figsize=(6, 1 * num_rows))

    for i in range(num_samples):
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row) * samples_per_row + i + 1)
        plt.title(f'{metrics[i]:.2f}')
        plt.imshow(probs[i])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row + 1) * samples_per_row + i + 1)
        plt.imshow(gt[i])
        plt.xticks([])
        plt.yticks([])
    plt.gcf().tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    image_logger(plt.gcf())

class PredictionGrid(Callback):
    def __init__(self, max_samples, image_logger, metric_fn):
        self.max_samples = max_samples
        self.image_logger = image_logger
        self.metric_fn = metric_fn

    def on_validation_end(self, logs, outputs, gt):
        visualize_predictions(self.image_logger, self.max_samples, self.metric_fn, outputs, gt)

