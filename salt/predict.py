import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from salt.model_checkpoint import load_checkpoint
from salt.generators import get_test_generator
from salt.utils import as_cuda
from salt.utils import encode_rle
from salt.utils import from_numpy
from salt.utils import get_images_in
from salt.utils import resize
from salt.utils import to_numpy

def predict(checkpoint_path, batch_size=1, limit=None):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    torch.set_grad_enabled(False)
    model.eval()

    records = []
    ids = list(map(lambda path: path.split('/')[-1].split('.')[0], get_images_in('data/test/images')))[:limit]
    test_generator = get_test_generator(batch_size, limit)
    for inputs, _ in tqdm(test_generator, total=len(test_generator)):
        inputs = from_numpy(inputs)
        mask_logits, image_logits = model(inputs)
        masks = to_numpy(torch.argmax(mask_logits, dim=1))
        image_logits = to_numpy(image_logits)
        image_logits -= np.expand_dims(np.max(image_logits, axis=1), axis=1)
        image_probs = (np.exp(image_logits) / np.expand_dims(np.sum(np.exp(image_logits), axis=1), axis=1))[:, 1]
        for mask, prob in zip(masks, image_probs):
            _id = ids.pop(0)
            if prob > 0.8:
                records.append((_id, None))
            else:
                records.append((_id, encode_rle(resize((101, 101), mask))))

    image_ids, encoded_pixels = zip(*records)
    df = pd.DataFrame({'id': image_ids, 'rle_mask': encoded_pixels})
    df.to_csv('./data/submissions/__latest.csv', index=False)

if __name__ == '__main__':
    Fire(predict)
