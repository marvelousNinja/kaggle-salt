import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def get_train_validation_holdout_split(records):
    np.random.seed(1991)
    np.random.shuffle(records)
    n = len(records)
    train = records[:int(n * .9)]
    validation = records[int(n * .9):int(n * 1.0)]
    holdout = records[int(n * 1.0):]
    return train, validation, holdout

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.png'))

def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def fliplr(image):
    return cv2.flip(image, 1)

def read_image_cached(cache, preprocess, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = preprocess(read_image(path))
        cache[path] = image
        return image

def normalize(image):
    return (image.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

def channels_first(image):
    return np.moveaxis(image, 2, 0)

def encode_rle(mask):
    pixels = mask.T.flatten()
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

def adjust_brightness(brightness, image):
    return image * brightness

def shift_and_scale(top, bottom, left, right, image):
    original_shape = image.shape[:2]
    image = image[top:image.shape[0] - bottom, left:image.shape[1] - right]
    return resize(original_shape, image)

def shear(height_shift, width_shift, interpolation, image):
    height, width = image.shape[:2]

    src_perspective = np.array([
        [0, 0],          # top-left
        [width, 0],      # top-right
        [width, height], # bottom-right
        [0, height]      # bottom-left
    ], np.float32)

    dst_perspective = np.array([
        [width_shift, height_shift],                    # top-left
        [width + width_shift, -height_shift],           # top-right
        [width - width_shift, height - height_shift],   # bottom-right
        [-width_shift, height + height_shift]           # bottom-left
    ], np.float32)

    return cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(src_perspective, dst_perspective),
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101
    )

def rotate(angle, interpolation, image):
    height, width = image.shape[:2]

    return cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1),
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101
    )

def reflect_pad(top, bottom, left, right, image):
    return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)

def blur(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def train_pipeline(mask_db, cache, mask_cache, path):
    preprocess = lambda image: image[:, :, [0]]
    image = read_image_cached(cache, preprocess, path)
    image = image[:, :, [0, 0, 0]]
    image = normalize(image)
    preprocess = lambda mask: mask
    mask = load_mask_cached(mask_cache, preprocess, mask_db, (101, 101), path)

    if np.random.rand() < .5:
        image = fliplr(image)
        mask = fliplr(mask)

    if np.random.rand() < .5:
        image = adjust_brightness(np.random.uniform(-0.2, 0.2) + 1, image)

    if np.random.rand() < .5:
        top, bottom = np.random.randint(0, image.shape[0] * 0.2, size=2)
        left, right = np.random.randint(0, image.shape[1] * 0.2, size=2)
        image = shift_and_scale(top, bottom, left, right, image)
        mask = shift_and_scale(top, bottom, left, right, mask)

    if np.random.rand() < .5:
        if np.random.rand() < .5:
            height_shift = np.random.randint(-image.shape[0] * 0.2, image.shape[0] * 0.2)
            width_shift = 0
        else:
            height_shift = 0
            width_shift = np.random.randint(-image.shape[1] * 0.2, image.shape[1] * 0.2)

        image = shear(height_shift, width_shift, cv2.INTER_LINEAR, image)
        mask = shear(height_shift, width_shift, cv2.INTER_NEAREST, mask)

    image = reflect_pad(13, 14, 13, 14, image)
    mask = reflect_pad(13, 14, 13, 14, mask)

    image = channels_first(image)
    return image, mask

def validation_pipeline(mask_db, cache, mask_cache, path):
    preprocess = lambda image: image[:, :, [0]]
    image = read_image_cached(cache, preprocess, path)
    image = image[:, :, [0, 0, 0]]
    image = normalize(image)
    preprocess = lambda mask: mask
    mask = load_mask_cached(mask_cache, preprocess, mask_db, (101, 101), path)
    image = reflect_pad(13, 14, 13, 14, image)
    mask = reflect_pad(13, 14, 13, 14, mask)
    image = channels_first(image)
    return image, mask

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
