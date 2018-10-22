import cv2
import numpy as np
from albumentations import (Blur, Compose, ElasticTransform, GaussNoise,
                            GridDistortion, HorizontalFlip, MedianBlur,
                            MotionBlur, Normalize, OneOf, PadIfNeeded,
                            RandomBrightness, RandomGamma, RandomSizedCrop,
                            ShiftScaleRotate)

from salt.utils import load_mask, read_image


def channels_first(image):
    return np.moveaxis(image, 2, 0)

def resize(size, interpolation, image):
    return cv2.resize(image, size, interpolation=interpolation)

def read_image_and_mask(mask_db, target_shape, path):
    image = read_image(path)
    mask = load_mask(mask_db, image.shape[:2], path)
    image = resize(target_shape, cv2.INTER_LINEAR, image)
    if mask is not None:
        mask = resize(target_shape, cv2.INTER_NEAREST, mask)
    return image, mask

def read_image_and_mask_cached(cache, mask_db, target_shape, path):
    if cache.get(path): return cache[path]
    image, mask = read_image_and_mask(mask_db, target_shape, path)
    cache[path] = (image, mask)
    return image, mask

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        args['mask'] = channels_first(args['mask'])
        return args

class LabelMaskBorder:
    def __call__(self, **args):
        mask = args.get('mask')
        if mask is None: return args
        kernel = np.ones((5, 5), np.uint8)
        borders = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        args['mask'] = np.dstack([mask, borders])
        return args

def train_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (101, 101), path)
    args = Compose([
        LabelMaskBorder(),
        HorizontalFlip(p=0.5),
        OneOf([
            ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_REPLICATE),
            RandomSizedCrop(min_max_height=(70, 100), height=101, width=101)
        ], p=0.2),
        GaussNoise(p=0.2),
        OneOf([
            RandomBrightness(limit=0.4),
            RandomGamma(),
        ], p=0.5),
        OneOf([
            Blur(),
            MedianBlur(),
            MotionBlur()
        ], p=0.2),
        OneOf([
            ElasticTransform(alpha=10, sigma=10, alpha_affine=10),
            GridDistortion()
        ], p=0.2),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        PadIfNeeded(128, 128, cv2.BORDER_REPLICATE),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return args['image'], args.get('mask')

def validation_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (101, 101), path)
    args = Compose([
        LabelMaskBorder(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        PadIfNeeded(128, 128, cv2.BORDER_REPLICATE),
        ChannelsFirst()
    ])(image=image, mask=mask)
    return args['image'], args.get('mask')
