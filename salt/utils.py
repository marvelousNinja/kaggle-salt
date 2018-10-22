import glob

import cv2
import numpy as np
import pandas as pd
import torch

def get_area_stratified_split(mask_db, num_folds):
    blacklist = [
        '05b69f83bf',
        '0d8ed16206',
        '10833853b3',
        '135ae076e9',
        '1a7f8bd454',
        '1b0d74b359',
        '1c0b2ceb2f',
        '1efe1909ed',
        '1f0b16aa13',
        '1f73caa937',
        '20ed65cbf8',
        '287b0f197f',
        '2fb6791298',
        '37df75f3a2',
        '3ee4de57f8',
        '3ff3881428',
        '40ccdfe09d',
        '423ae1a09c',
        '4f30a97219',
        '51870e8500',
        '573f9f58c4',
        '58789490d6',
        '590f7ae6e7',
        '5aa0015d15',
        '5edb37f5a8',
        '5ff89814f5',
        '6b95bc6c5f',
        '6f79e6d54b',
        '755c1e849f',
        '762f01c185',
        '7769e240f0',
        '808cbefd71',
        '8c1d0929a2',
        '8ee20f502e',
        '9260b4f758',
        '96049af037',
        '96d1d6138a',
        '97515a958d',
        '99909324ed',
        '9aa65d393a',
        'a2b7af2907',
        'a31e485287',
        'a3e0a0c779',
        'a48b9989ac',
        'a536f382ec',
        'a56e87840f',
        'a8be31a3c1',
        'a9e940dccd',
        'a9fd8e2a06',
        'aa97ecda8e',
        'acb95dd7c9',
        'b11110b854',
        'b552fb0d9d',
        'b637a7621a',
        'b8c3ca0fab',
        'b9bf0422a6',
        'bedb558d15',
        'c1c6a1ebad',
        'c20069b110',
        'c3589905df',
        'c8404c2d4f',
        'cc15d94784',
        'd0244d6c38',
        'd0e720b57b',
        'd1665744c3',
        'd2e14828d5',
        'd6437d0c25',
        'd8bed49320',
        'd93d713c55',
        'dcca025cc6',
        'e0da89ce88',
        'e51599adb5',
        'e7da2d7800',
        'e82421363e',
        'ec542d0719',
        'f0190fc4b4',
        'f26e6cffd6',
        'f2c869e655',
        'f9fc7746fb',
        'ff9d2e9ba7',
    ]

    mask_db = mask_db[~mask_db['id'].isin(blacklist)].copy()
    np.random.seed(1991)
    mask_db['fold_id'] = np.random.randint(0, num_folds, len(mask_db))
    return mask_db['id'].values, mask_db['fold_id'].values

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
