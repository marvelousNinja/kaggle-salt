import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate

def get_train_validation_holdout_split(records):
    np.random.shuffle(records)
    n = len(records)
    train = records[:int(n * .6)]
    validation = records[int(n * .6):int(n * .75)]
    holdout = records[int(n * .75)]
    return train, validation, holdout

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.png'))

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def fliplr(image):
    return np.fliplr(image)

def read_image_cached(cache, preprocess, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = preprocess(read_image(path))
        cache[path] = image
        return image

def normalize(image):
    return (image.astype(np.float32) / 255 - [0.471]) / [0.108]

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def encode_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle(shape, encoded_mask):
    if encoded_mask == 'nan': return np.zeros(shape)
    numbers = np.array(list(map(int, encoded_mask.split())))
    starts, lengths = numbers[::2], numbers[1::2]
    # Enumerates from 1
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends): mask[start:end] += 1
    mask = np.clip(mask, a_min=0, a_max=2)
    return mask.reshape(shape).T

def load_mask(mask_db, shape, image_path):
    image_id = image_path.split('/')[-1].split('.')[0]
    labelled_mask = np.zeros(shape)

    for i, encoded_mask in enumerate(mask_db[mask_db['id'] == image_id]['rle_mask'].fillna('nan')):
        labelled_mask[decode_rle(shape, encoded_mask) == 1] = i + 1
    return labelled_mask.astype(np.uint8)

def load_mask_cached(cache, preprocess, mask_db, shape, path):
    mask = cache.get(path)
    if mask is not None:
        return mask
    else:
        mask = preprocess(load_mask(mask_db, shape, path))
        cache[path] = mask
        return mask

def resize(size, image):
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

def pipeline(mask_db, cache, mask_cache, path):
    preprocess = lambda image: resize((128, 128), image[:, :, [1]])[:, :, None]
    image = read_image_cached(cache, preprocess, path)
    image = normalize(image)
    image = channels_first(image)
    preprocess = lambda mask: resize((128, 128), mask)
    mask = load_mask_cached(mask_cache, preprocess, mask_db, (101, 101), path)
    if np.random.rand() < .5:
        image = fliplr(image)
        mask = fliplr(mask)
    return image, mask

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

def visualize_predictions(image_logger, logits, gt):
    num_samples = min(len(gt), 8)
    gt = gt[:num_samples]
    logits = logits[:num_samples]
    logits -= np.expand_dims(np.max(logits, axis=1), axis=1)
    probs = (np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), axis=1), axis=1))[:, 1, :, :]

    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(probs[i])
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(gt[i])
    plt.gcf().tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    image_logger(plt.gcf())

def visualize_learning_curve(image_logger, train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(np.array(train_losses) - np.array(val_losses), label='Generalization Error')
    plt.legend()
    image_logger(plt.gcf())

def get_mask_db(path):
    return pd.read_csv(path)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(obj)
    else:
        return torch.FloatTensor(obj)

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

if __name__ == '__main__':
    import pdb; pdb.set_trace()
