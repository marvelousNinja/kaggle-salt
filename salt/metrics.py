import torch

def mean_iou(outputs, gt):
    smooth = 1e-12
    pred_masks = torch.argmax(outputs, dim=1)
    true_masks = gt.long()
    intersection = (pred_masks & true_masks).sum(dim=(1, 2)).float()
    union = (pred_masks | true_masks).sum(dim=(1, 2)).float()
    return ((intersection + smooth) / (union + smooth)).mean()
