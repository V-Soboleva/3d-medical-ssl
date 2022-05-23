import typing as tp
import torch
from torch import nn
import torch.nn.functional as F

from .utils import Transform2D


def compute_pixelwise_loss_3d(
        image: torch.Tensor,
        model: nn.Module,
        transforms: tp.Sequence[Transform2D],
        temperature: float,
        min_neg_distance_vxl: float,
        roi: tp.Optional[torch.Tensor] = None
):
    """Transformation loss.

    Args:
        image (torch.Tensor): tensor of shape (c, h, w, d).
        model (nn.Module): FCN.
        transforms (Sequence[Transform2D]): sequence of applied transforms.
        min_neg_distance_pxl (float): features located at a distance
            more than ``min_neg_distance_pxl`` are treated as negative pair.
    """
    if roi is None:
        roi = torch.ones(image.shape[1:], dtype=bool)

    assert roi.shape == image.shape[1:]

    fmaps = model(torch.stack((image, *[t(image) for t in transforms])))
    fmap, fmaps = fmaps[0], fmaps[1:]

    rois = [t(roi) for t in transforms]

    _grid = torch.stack(torch.meshgrid(*map(torch.arange, image.shape[1:]))).to(dtype=torch.float32)  # (3, h, w, d)
    grids = [t(_grid, fill_value=float('nan')) for t in transforms]

    loss = torch.tensor(0., requires_grad=True).to(fmaps)
    cnt = 0
    for idx, transform in enumerate(transforms):
        transformed_fmap = transform(fmap)  # (c, h, w, d)

        neg_idx = [i for i in range(len(transforms)) if i != idx]
        neg_mask = torch.stack([
            rois[i] & (torch.norm(grids[i] - grids[idx], dim=0) >= min_neg_distance_vxl)
            for i in neg_idx
        ])  # (n - 1, h, w, d)

        mask = rois[idx] & torch.any(neg_mask, dim=0)

        if mask.any():
            loss = loss + _compute_triplet_loss(
                anchor_fmap=transformed_fmap[:, mask],  # (c, m)
                pos_fmap=fmaps[idx][:, mask],  # (c, m)
                neg_fmaps=fmaps[neg_idx][:, :, mask],  # (n - 1, c, m)
                temperature=temperature,
                neg_mask=neg_mask[:, mask]  # (n - 1, m)
            )
            cnt += 1

    if cnt:
        loss = loss / cnt

    return loss


def _compute_triplet_loss(
        anchor_fmap: torch.Tensor,
        pos_fmap: torch.Tensor,
        neg_fmaps: torch.Tensor,
        temperature: float,
        neg_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Calculates contrastive loss.

    Args:
        fmap (torch.Tensor): (c, ...).
        pos_fmap (torch.Tensor): (c, ...).
        neg_fmaps (torch.Tensor): (n, c, ...).
        temperature (float): > 0.
        neg_mask (torch.Tensor): (n, ...).

    Returns:
        torch.Tensor: scalar.
    """
    pos_sim = torch.cosine_similarity(anchor_fmap, pos_fmap, dim=0).unsqueeze(0)  # (1, m)
    neg_sim = torch.cosine_similarity(anchor_fmap.unsqueeze(0), neg_fmaps, dim=1)  # (n, m)
    if neg_mask is not None:
        neg_mask = neg_mask.to(device=neg_sim.device)
        neg_sim = torch.where(neg_mask, neg_sim, torch.tensor(float('-inf')).to(neg_sim))
    cross_entropy = -torch.log_softmax(torch.cat((pos_sim, neg_sim)) / temperature, dim=0)[0]  # (m, d)
    return torch.mean(cross_entropy)
