import torch

def flat_lovasz_hinge_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[0:-1]
    return jaccard

def flat_lovasz_hinge_loss(logits, labels):
    signs = 2. * labels.float() - 1.
    errors = 1 - signs * logits
    errors_sorted, indicies = torch.sort(errors, dim=0, descending=True)
    labels_sorted = labels[indicies]
    grad = flat_lovasz_hinge_grad(labels_sorted)
    return torch.dot(torch.nn.functional.relu(errors_sorted), grad)

def lovasz_hinge_loss(logits, labels, average=True):
    losses = []
    for sample_logits, sample_labels in zip(logits, labels):
        losses.append(flat_lovasz_hinge_loss(sample_logits.view(-1), sample_labels.view(-1)))
    if average: return sum(losses) / len(losses)
    return losses
