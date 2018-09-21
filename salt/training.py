import numpy as np
import torch
import torchvision
from tabulate import tabulate
from tqdm import tqdm

from salt.utils import from_numpy
from salt.utils import to_numpy
from salt.utils import as_cuda

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs,
        logger,
        callbacks=[],
        metrics=[]
    ):

    advesary_model = torchvision.models.resnet18(pretrained=True)
    advesary_model.avgpool = torch.nn.AvgPool2d(4)
    advesary_model.fc = torch.nn.Linear(advesary_model.fc.in_features, 1)
    advesary_model = as_cuda(advesary_model)
    advesary_opt = torch.optim.SGD(advesary_model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        num_batches = len(train_generator)
        # 1. Training advesary net
        torch.set_grad_enabled(True)
        model.eval()
        advesary_model.train()
        advesary_loss = 0
        for inputs, gt in tqdm(train_generator, total=num_batches):
            # 1.1. Training on normal batch
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            advesary_opt.zero_grad()
            outputs = advesary_model(inputs * gt[:, None, :, :])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, as_cuda(torch.ones(gt.shape[0])[:, None]))
            loss.backward()
            advesary_opt.step()
            advesary_loss += loss.data[0]
            # 1.2. Training on fake batch
            segmentator_preds = model(inputs).detach()
            advesary_opt.zero_grad()
            outputs = advesary_model(inputs * segmentator_preds)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, as_cuda(torch.zeros(gt.shape[0])[:, None]))
            loss.backward()
            advesary_opt.step()
            advesary_loss += loss.data[0]
        logger(f'Advesary loss: {advesary_loss / (num_batches * 2):.5f}')

        logs = {}
        logs['train_loss'] = 0
        for func in metrics: logs[f'train_{func.__name__}'] = 0
        num_batches = len(train_generator)
        model.train()
        advesary_model.eval()
        torch.set_grad_enabled(True)
        for callback in callbacks: callback.on_train_begin()
        for inputs, gt in tqdm(train_generator, total=num_batches):
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            optimizer.zero_grad()
            outputs = model(inputs)
            advesary_outputs = advesary_model(inputs * outputs)
            loss = loss_fn(outputs, gt) + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(advesary_outputs, as_cuda(torch.ones(gt.shape[0])[:, None]))
            loss.backward()
            optimizer.step()
            logs['train_loss'] += loss.data[0]
            for func in metrics: logs[f'train_{func.__name__}'] += func(outputs.detach(), gt)
            for callback in callbacks: callback.on_train_batch_end(loss.data[0])

        logs['train_loss'] /= num_batches
        for func in metrics: logs[f'train_{func.__name__}'] /= num_batches

        logs['val_loss'] = 0
        for func in metrics: logs[f'val_{func.__name__}'] = 0
        all_outputs = []
        all_gt = []
        num_batches = len(validation_generator)
        model.eval()
        torch.set_grad_enabled(False)
        for inputs, gt in tqdm(validation_generator, total=num_batches):
            all_gt.append(gt)
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            outputs = model(inputs)
            logs['val_loss'] += loss_fn(outputs, gt).data[0]
            for func in metrics: logs[f'val_{func.__name__}'] += func(outputs.detach(), gt)

            if isinstance(outputs, tuple):
                all_outputs.append(list(map(to_numpy, outputs)))
            else:
                all_outputs.append(to_numpy(outputs))
        logs['val_loss'] /= num_batches
        for func in metrics: logs[f'val_{func.__name__}'] /= num_batches

        if isinstance(all_outputs[0], tuple):
            all_outputs = list(map(np.concatenate, zip(*all_outputs)))
        else:
            all_outputs = np.concatenate(all_outputs)

        all_gt = np.concatenate(all_gt)
        for callback in callbacks: callback.on_validation_end(logs, all_outputs, all_gt)

        epoch_rows = [['epoch', epoch]]
        for name, value in logs.items():
            epoch_rows.append([name, f'{value:.3f}'])

        logger(tabulate(epoch_rows))
