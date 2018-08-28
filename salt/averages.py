import numpy as np
from salt.utils import read_image
from salt.utils import get_images_in
from tqdm import tqdm

image_mean_values = []
image_std_values = []
image_paths = get_images_in('data/train/images')
for path in tqdm(image_paths):
    image = read_image(path)
    image = image[:, :, [1]] / 255
    image_mean_values.append(image.mean())
    image_std_values.append(image.std())

print(np.mean(image_mean_values))
print(np.mean(image_std_values))
import pdb; pdb.set_trace()
