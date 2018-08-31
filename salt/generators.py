import math
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np

from salt.utils import get_images_in
from salt.utils import get_mask_db
from salt.utils import get_train_validation_holdout_split
from salt.utils import train_pipeline
from salt.utils import validation_pipeline

class DataGenerator:
    def __init__(self, records, batch_size, transform, shuffle=True):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.records)
        batch = []
        pool = ThreadPool()
        prefetch_size = 2000
        num_slices = len(self.records) // prefetch_size + 1

        for i in range(num_slices):
            start = i * prefetch_size
            end = start + prefetch_size
            for output in pool.imap(self.transform, self.records[start:end]):
                batch.append(output)
                if len(batch) >= self.batch_size:
                    split_outputs = list(zip(*batch))
                    yield list(map(np.stack, split_outputs))
                    batch = []

        if len(batch) > 0:
            split_outputs = list(zip(*batch))
            yield list(map(np.stack, split_outputs))

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

def get_validation_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    image_paths = get_images_in('data/train/images')
    _, image_paths, _ = get_train_validation_holdout_split(image_paths)
    transform = partial(validation_pipeline, mask_db, {}, {})
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False)

def get_train_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    image_paths = get_images_in('data/train/images')
    image_paths, _, _ = get_train_validation_holdout_split(image_paths)
    transform = partial(train_pipeline, mask_db, {}, {})
    return DataGenerator(image_paths[:limit], batch_size, transform)

def get_test_generator(batch_size, limit=None):
    mask_db = get_mask_db('data/train.csv')
    image_paths = get_images_in('data/test/images')
    transform = partial(validation_pipeline, mask_db, {}, {})
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False)
