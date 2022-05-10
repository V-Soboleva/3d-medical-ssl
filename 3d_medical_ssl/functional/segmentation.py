import torch


def compute_dice_loss(probas, masks, spatial_dims):
    intersection = torch.sum(probas * masks, dim=spatial_dims)
    volumes_sum = torch.sum(probas ** 2 + masks ** 2, dim=spatial_dims)
    dice = 2 * intersection / (volumes_sum + 1)
    loss = 1 - dice
    return loss.mean()
