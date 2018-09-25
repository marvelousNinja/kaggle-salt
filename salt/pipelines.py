import cv2
import numpy as np

from salt.utils import read_image
from salt.utils import load_mask

def fliplr(image):
    return cv2.flip(image, 1)

def normalize(image):
    return (image.astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

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

def adjust_brightness(brightness, image):
    return image * brightness

def shift_and_scale(interpolation, top, bottom, left, right, image):
    original_shape = image.shape[:2]
    image = image[top:image.shape[0] - bottom, left:image.shape[1] - right]
    return resize(original_shape, interpolation, image)

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

def edge_pad(top, bottom, left, right, image):
    return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

class Pipe:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, args):
        for func in self.funcs: args = func(args)
        return args

class OneOf:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, args):
        return np.random.choice(self.funcs)(args)

class Maybe:
    def __init__(self, p, func):
        self.p = p
        self.func = func

    def __call__(self, args):
        if np.random.rand() < self.p: return self.func(args)
        return args

class AdjustBrightness:
    def __init__(self, diff):
        self.diff = diff

    def __call__(self, args):
        adjustment = np.random.uniform(-self.diff, self.diff) + 1
        args['image'] = adjust_brightness(adjustment, args['image'])
        return args

class VerticalShear:
    def __init__(self, diff):
        self.diff = diff

    def __call__(self, args):
        image = args['image']
        shift = np.random.randint(-image.shape[0] * self.diff, image.shape[0] * self.diff)
        args['image'] = shear(shift, 0, cv2.INTER_LINEAR, image)
        if args.get('mask') is not None:
            args['mask'] = shear(shift, 0, cv2.INTER_NEAREST, args['mask'])
        return args

class HorizontalShear:
    def __init__(self, diff):
        self.diff = diff

    def __call__(self, args):
        image = args['image']
        shift = np.random.randint(-image.shape[1] * self.diff, image.shape[1] * self.diff)
        args['image'] = shear(0, shift, cv2.INTER_LINEAR, image)
        if args.get('mask') is not None:
            args['mask'] = shear(0, shift, cv2.INTER_NEAREST, args['mask'])
        return args

class ShiftScale:
    def __init__(self, diff):
        self.diff = diff

    def __call__(self, args):
        image = args['image']
        top, bottom = np.random.randint(0, image.shape[0] * self.diff, size=2)
        left, right = np.random.randint(0, image.shape[1] * self.diff, size=2)
        args['image'] = shift_and_scale(cv2.INTER_LINEAR, top, bottom, left, right, image)
        if args.get('mask') is not None:
            args['mask'] = shift_and_scale(cv2.INTER_NEAREST, top, bottom, left, right, args['mask'])
        return args

class Rotate:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, args):
        angle = np.random.randint(-self.max_angle, self.max_angle)
        args['image'] = rotate(angle, cv2.INTER_LINEAR, args['image'])
        if args.get('mask') is not None:
            args['mask'] = rotate(angle, cv2.INTER_NEAREST, args['mask'])
        return args

class Fliplr:
    def __call__(self, args):
        args['image'] = fliplr(args['image'])
        if args.get('mask') is not None:
            args['mask'] = fliplr(args['mask'])
        return args

class EdgePad:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, args):
        args['image'] = edge_pad(self.top, self.bottom, self.left, self.right, args['image'])
        if args.get('mask') is not None:
            args['mask'] = edge_pad(self.top, self.bottom, self.left, self.right, args['mask'])
        return args

class Normalize:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, args):
        args['image'] = (args['image'].astype(np.float32) / 255 - self.means) / self.stds
        return args

class ChannelsFirst:
    def __call__(self, args):
        args['image'] = channels_first(args['image'])
        return args

class Blur:
    def __call__(self, args):
        args['image'] = blur(args['image'])
        return args

class Cutout:
    def __init__(self, diff):
        self.diff = diff

    def __call__(self, args):
        image = args['image'].copy()
        cut_height = int(image.shape[0] * self.diff)
        cut_width = int(image.shape[1] * self.diff)
        top = np.random.randint(0, cut_height)
        left = np.random.randint(0, cut_width)
        image[top:top + cut_height, left:left + cut_width] = 0
        args['image'] = image
        if args.get('mask') is not None:
            args['mask'] = args['mask'].copy()
            args['mask'][top:top + cut_height, left:left + cut_width] = 0
        return args

class CloseMask:
    def __call__(self, args):
        kernel = np.ones((4, 4), np.uint8)
        if args.get('mask') is not None:
            args['mask'] = cv2.morphologyEx(args.get('mask'), cv2.MORPH_CLOSE, kernel)
        return args

def train_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (101, 101), path)
    args = Pipe([
        Maybe(0.5, Fliplr()),
        Maybe(0.5, OneOf([
            AdjustBrightness(0.2),
            VerticalShear(0.2),
            HorizontalShear(0.2),
            ShiftScale(0.2),
            Rotate(10),
            Cutout(0.2),
            Blur()
        ])),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        EdgePad(13, 14, 13, 14),
        ChannelsFirst()
    ])({'image': image, 'mask': mask})
    return args['image'], args.get('mask')

def validation_pipeline(cache, mask_db, path):
    image, mask = read_image_and_mask_cached(cache, mask_db, (101, 101), path)
    args = Pipe([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        EdgePad(13, 14, 13, 14),
        ChannelsFirst()
    ])({'image': image, 'mask': mask})
    return args['image'], args.get('mask')
