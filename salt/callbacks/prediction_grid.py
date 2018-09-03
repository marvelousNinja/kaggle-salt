import matplotlib.pyplot as plt
import numpy as np

from salt.callbacks.callback import Callback

def visualize_predictions(image_logger, max_samples, logits, gt):
    num_samples = min(len(gt), max_samples)
    gt = gt[:num_samples]
    logits = logits[:num_samples]
    logits -= np.expand_dims(np.max(logits, axis=1), axis=1)
    probs = (np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), axis=1), axis=1))[:, 1, :, :]

    samples_per_row = 8
    num_rows = int(np.ceil(num_samples / samples_per_row)) * 2
    plt.figure(figsize=(6, 1 * num_rows))

    for i in range(num_samples):
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row) * samples_per_row + i + 1)
        plt.imshow(probs[i])
        plt.subplot(num_rows, samples_per_row, (i // samples_per_row + 1) * samples_per_row + i + 1)
        plt.imshow(gt[i])
    plt.gcf().tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    image_logger(plt.gcf())

class PredictionGrid(Callback):
    def __init__(self, max_samples, image_logger):
        self.max_samples = max_samples
        self.image_logger = image_logger

    def on_validation_end(self, logs, outputs, gt):
        visualize_predictions(self.image_logger, self.max_samples, outputs, gt)

