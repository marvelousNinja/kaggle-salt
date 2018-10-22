import numpy as np
import pandas as pd
import torch
from fire import Fire
from tqdm import tqdm

from salt.callbacks.model_checkpoint import load_checkpoint
from salt.generators import get_validation_generator
from salt.losses import lovasz_hinge_loss
from salt.utils import as_cuda
from salt.utils import from_numpy
from salt.utils import get_images_in
from salt.utils import get_mask_db
from salt.utils import get_area_stratified_split
from salt.utils import to_numpy

def find_outliers(checkpoint_path, num_folds=5, fold_ids=[0, 1, 2, 3], batch_size=1, limit=None, tta=False):
    model = load_checkpoint(checkpoint_path)
    model = as_cuda(model)
    torch.set_grad_enabled(False)
    model.eval()

    mask_db = get_mask_db('data/train.csv')
    all_image_ids, all_fold_ids = get_area_stratified_split(mask_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]

    losses = []
    generator = get_validation_generator(num_folds, fold_ids, batch_size, limit)
    for inputs, gt in tqdm(generator, total=len(generator)):
        inputs, gt = from_numpy(inputs), from_numpy(gt)
        outputs = model(inputs)
        if tta:
            flipped_outputs = model(inputs.flip(dims=(3,)))
            outputs = (outputs + flipped_outputs.flip(dims=(3,))) / 2
        batch_losses = to_numpy(lovasz_hinge_loss(outputs, gt, average=False))
        losses.extend(batch_losses)

    image_ids = image_ids[:len(losses)]
    import pdb; pdb.set_trace()
    stats = pd.DataFrame(data={'image_id': image_ids, 'loss': losses})
    print(stats.sort_values('loss'))


if __name__ == '__main__':
    Fire(find_outliers)
