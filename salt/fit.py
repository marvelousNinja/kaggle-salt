import torch
import numpy as np
import matplotlib; matplotlib.use('agg')
from fire import Fire

from salt.callbacks.cyclic_lr import CyclicLR
from salt.callbacks.confusion_matrix import ConfusionMatrix
from salt.callbacks.learning_curve import LearningCurve
from salt.callbacks.loss_surface import LossSurface
from salt.callbacks.lr_on_plateau import LROnPlateau
from salt.callbacks.lr_range_test import LRRangeTest
from salt.callbacks.lr_schedule import LRSchedule
from salt.callbacks.histogram import Histogram
from salt.callbacks.model_checkpoint import ModelCheckpoint
from salt.callbacks.model_checkpoint import load_checkpoint
from salt.callbacks.prediction_grid import PredictionGrid
from salt.callbacks.weight_grid import WeightGrid
from salt.generators import get_train_generator
from salt.generators import get_validation_generator
from salt.loggers import make_loggers
from salt.losses import lovasz_hinge_loss
from salt.losses import focal_loss
from salt.metrics import mean_iou
from salt.metrics import mean_ap
from salt.models.devilnet import Devilnet
from salt.training import fit_model
from salt.utils import as_cuda

def loss_surface_fn(outputs, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), labels, reduction='none')

def compute_loss(outputs, labels):
    true_borders = labels[:, 1, :, :]
    pred_borders = outputs[:, 1, :, :]
    border_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_borders, true_borders)
    mask_present = (true_borders.sum(dim=(1, 2)) > 0).nonzero().view(-1)
    outputs, labels = outputs[mask_present], labels[mask_present]
    if len(outputs) > 0:
        mask_loss = lovasz_hinge_loss(outputs[:, [0], :, :], labels[:, 0, :, :])
    else:
        mask_loss = 0
    return 10 * border_loss + mask_loss

def fit(
        num_epochs=100,
        limit=None,
        validation_limit=None,
        batch_size=16,
        lr=.005,
        checkpoint_path=None,
        telegram=False,
        visualize=False,
        num_folds=5,
        train_fold_ids=[0, 1, 2, 3],
        validation_fold_ids=[4]
    ):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    logger, image_logger = make_loggers(telegram)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Devilnet(2)

    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-3, momentum=0.9, nesterov=True)
    train_generator = get_train_generator(num_folds, train_fold_ids, batch_size, limit)
    callbacks = [
        ModelCheckpoint(model, 'linknet', 'val_mean_ap', 'max', logger),
        # CyclicLR(step_size=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger),
        # LRSchedule(optimizer, [(0, 0.003), (2, 0.01), (12, 0.001), (17, 0.0001)], logger),
        # LRRangeTest(0.00001, 1.0, 20000, optimizer, image_logger),
        LROnPlateau('val_mean_ap', optimizer, mode='max', factor=0.5, patience=8, min_lr=0, logger=logger),
        # ConfusionMatrix([0, 1], logger)
    ]

    if visualize:
        callbacks.extend([
            LearningCurve(['train_loss', 'val_loss', 'train_mean_iou', 'val_mean_iou', 'train_mean_ap', 'val_mean_ap'], image_logger),
            PredictionGrid(80, image_logger, mean_iou),
            LossSurface(image_logger, loss_surface_fn),
            Histogram(image_logger, mean_iou),
            WeightGrid(model, image_logger, 32)
        ])

    fit_model(
        model=model,
        train_generator=train_generator,
        validation_generator=get_validation_generator(num_folds, validation_fold_ids, batch_size, validation_limit),
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
