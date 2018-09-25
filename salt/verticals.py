import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from salt.utils import decode_rle

vertical_masks = []
mask_db = pd.read_csv('./data/train.csv')
for _, row in tqdm(mask_db.iterrows()):
    id = row['id']
    if str(row['rle_mask']) == 'nan': continue
    mask = decode_rle((101, 101), row['rle_mask'])
    mask_area = mask.sum()
    if mask_area > 0 and mask_area % 101 == 0:
        image = cv2.imread(f'data/train/images/{id}.png')
        plt.imshow(image)
        plt.imshow(mask, alpha=0.3)
        plt.show()
        vertical_masks.append(id)

print(len(vertical_masks))
print(vertical_masks)


blacklist = ['a266a2a9df', '50d3073821', 'd4d34af4f7', '7845115d01', '7deaf30c4a', '28553d5d42', 'b525824dfc', 'ab18a0a7fa', '483b35d589', 'f0c401b64b', '849881c690', 'f9b7196dfc', 'a9ee40cf0d', 'c83d9529bd', '05be526826', '52667992f8', 'ba1287cb48', 'aeba5383e4', 'caccd6708f', 'cb4f7abe67', 'd0bbe4fd97', '9b29ca561d', '3cb59a4fdc', 'b24d3673e1', 'b2c6b8cf57', 'f6e87c1458', '80a458a2b6', '6bc4c91c27', 'c387a012fc', '09b9330300', '8367b54eac', '95f6e2b2d1', '09152018c4', 'be18a24c49', '23afbccfb5', '58de316918', '62aad7556c', '24522ec665', 'c2973c16f1', '96523f824a', 'f641699848', '0280deb8ae', 'd9a52dc263', '50b3aef4c4', '39cd06da7d', '52ac7bb4c1', '7c0b76979f', 'd4d2ed6bd2', 'be7014887d', 'f75842e215', 'dd6a04d456', 'e335542c17', '62d30854d7', '33887a0ae7', 'c27409a765', '130229ec15', '7f0825a2f0', '1eaf42beee', 'e73ed6e7f2', '876e6423e6', 'fb3392fee0', '2424f4afc7', 'b35b1b412b', '0b45bde756', '4f5df40ab2', 'f7380099f6', '9a4b15919d', 'ddcb457a07', 'b8a9602e21', '87afd4b1ca', 'f19b7d20bb', 'e6e3e58c43', 'eeecc7ab39', '5b217529e7', '182bfc6862', 'baac3469ae', '4fdc882e4b', '33dfce3a76', 'a6625b8937', '834861f1b6', '2f746f8726', '40bb5a9cbe', '5f98029612', '3975043a11', '06d21d76c4', 'a54d582262', 'b63b23fdc9', 'bfbb9b9149', '3ce41108fe', '6460ce2df7', '4fbda008c7', '81fa3d59b8', '285f4b2e82', '9067effd34', '96216dae3b', 'b1be1fa682', '88a5c49514', '608567ed23', '99ee31b5bc', '71f7425387', '90720e8172', '2bc179b78c', '916aff36ae', '56f4bcc716', '00950d1627', 'febd1d2a67', '403cb8f4b3', '4ef0559016', '9eb4a10b98', 'bfa7ee102e', '93a1541218', '640ceb328a', 'b7b83447c4', 'c98dfd50ba', 'de7202d286', 'be90ab3e56', '15d76f1672', 'cef03959d8', '53e17edd83', 'e12cd094a6', '49336bb17b', 'ad2fa649f7', 'fb47e8e74e', '919bc0e2ba']
blacklist = list(map(lambda id: f'data/train/images/{id}.png'))
