import io
from functools import partial

import torch
import telegram_send
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from fire import Fire

from salt.callbacks.cyclic_lr import CyclicLR
from salt.callbacks.confusion_matrix import ConfusionMatrix
from salt.callbacks.learning_curve import LearningCurve
from salt.callbacks.loss_surface import LossSurface
from salt.callbacks.histogram import Histogram
from salt.callbacks.model_checkpoint import ModelCheckpoint
from salt.callbacks.model_checkpoint import load_checkpoint
from salt.callbacks.prediction_grid import PredictionGrid
from salt.generators import get_train_generator
from salt.generators import get_validation_generator
from salt.linknet import Linknet
from salt.loggers import make_loggers
from salt.metrics import mean_iou
from salt.metrics import mean_ap
from salt.training import fit_model
from salt.utils import as_cuda

def loss_surface_fn(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs, labels.long(), reduction='none')

def compute_loss(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs, labels.long())

def fit(num_epochs=100, limit=None, validation_limit=None, batch_size=16, lr=.005, checkpoint_path=None, telegram=False, visualize=False):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    logger, image_logger = make_loggers(telegram)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Linknet(2)

    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-4, momentum=0.9)
    train_generator = get_train_generator(batch_size, limit)
    callbacks = [
        ModelCheckpoint(model, 'linknet', 'val_mean_ap', 'max', logger),
        # CyclicLR(cycle_iterations=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger),
        ConfusionMatrix([0, 1], logger)
    ]

    if visualize:
        callbacks.extend([
            LearningCurve(['train_loss', 'val_loss', 'train_mean_iou', 'val_mean_iou', 'train_mean_ap', 'val_mean_ap'], image_logger),
            PredictionGrid(8, image_logger),
            LossSurface(image_logger, loss_surface_fn),
            Histogram(image_logger, mean_iou)
        ])

    fit_model(
        model=model,
        train_generator=train_generator,
        validation_generator=get_validation_generator(batch_size, validation_limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        metrics=[mean_iou, mean_ap]
    )

def prof():
    import profile
    import pstats
    profile.run('fit(batch_size=4, limit=100, num_epochs=1)', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
