import torch

def mean_iou(outputs, gt, average=True):
    smooth = 1e-12
    # TODO AS: Ignoring reflected regions, since they are cut on submission
    pred_masks = torch.sigmoid(outputs.squeeze()).round().long()[:, 13:-14, 13:-14]
    true_masks = gt.long()[:, 13:-14, 13:-14]
    intersection = (pred_masks & true_masks).sum(dim=(1, 2)).float()
    union = (pred_masks | true_masks).sum(dim=(1, 2)).float()
    values = ((intersection + smooth) / (union + smooth))
    return values.mean() if average else values

def mean_ap(outputs, gt, average=True):
    ious = mean_iou(outputs, gt, average=False)
    precision = 0
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds: precision += ious > threshold
    return (precision.float() / len(thresholds)).mean()
