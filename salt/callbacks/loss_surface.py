import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter

from salt.callbacks.callback import Callback
from salt.utils import to_numpy
from salt.utils import from_numpy

class LossSurface(Callback):
    def __init__(self, image_logger, loss_fn):
        self.image_logger = image_logger
        self.loss_fn = loss_fn

    def on_validation_end(self, logs, outputs, gt):
        losses = to_numpy(self.loss_fn(from_numpy(outputs), from_numpy(gt)))
        losses = losses.mean(axis=0)
        losses =  uniform_filter(losses, size=6, mode='nearest')
        x, y = np.meshgrid(np.arange(losses.shape[0]), np.arange(losses.shape[1]))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, losses, linewidth=0, antialiased=True, cmap=plt.get_cmap('viridis'), edgecolor='none')
        ax.view_init(60, 35)
        self.image_logger(fig)
