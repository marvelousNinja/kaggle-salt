import torch

def mean_iou(outputs, gt, average=True):
    smooth = 1e-12
    pred_masks = torch.argmax(outputs, dim=1)
    true_masks = gt.long()
    intersection = (pred_masks & true_masks).sum(dim=(1, 2)).float()
    union = (pred_masks | true_masks).sum(dim=(1, 2)).float()
    values = ((intersection + smooth) / (union + smooth))
    return values.mean() if average else values

def mean_ap(outputs, gt):
    ious = mean_iou(outputs, gt, average=False)
    precision = 0
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds: precision += ious > threshold
    precision /= len(thresholds)
    return precision.float().mean()
