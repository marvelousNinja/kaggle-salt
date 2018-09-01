import matplotlib.pyplot as plt

from salt.callbacks.callback import Callback
from salt.utils import to_numpy
from salt.utils import from_numpy

class Histogram(Callback):
    def __init__(self, image_logger, metric_fn):
        self.image_logger = image_logger
        self.metric_fn = metric_fn

    def on_validation_end(self, logs, outputs, gt):
        values = to_numpy(self.metric_fn(from_numpy(outputs), from_numpy(gt), average=False))
        plt.hist(values, bins=20)
        plt.title(self.metric_fn.__name__)
        self.image_logger(plt.gcf())
