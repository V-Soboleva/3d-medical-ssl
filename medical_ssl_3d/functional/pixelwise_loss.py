import typing as tp
import torch
from torch import nn
import torch.nn.functional as F

from medical_ssl_3d.functional import Transform2D


def pixelwise_loss(
        image: torch.Tensor,
        model: nn.Module,
        transform: tp.Sequence[Transform2D],
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

    tf = model(image.unsqueeze(0)).squeeze(0)
    tf = transform(tf)

    _grid = torch.stack(torch.meshgrid(*map(torch.arange, image.shape[1:]))).to(dtype=torch.float32)  # (3, h, w, d)
    grid = transform(_grid)
    
    ft = model(transform(image).unsqueeze(0)).squeeze(0)

    roi = transform(roi.unsqueeze(0)).squeeze(0)

    loss = torch.tensor(0., requires_grad=True).to(ft)

    neg_mask = (torch.norm(_grid - grid, dim=0) >= min_neg_distance_vxl) 
    neg_mask = neg_mask.to(device=tf.device)
    
    print(neg_mask.shape)
    
    similarity = torch.cosine_similarity(tf, ft, dim=0)
    
    print(similarity.shape)
    
    neg_sim = torch.where(neg_mask[roi], similarity[roi], torch.tensor(float('-inf')).to(similarity))
    cross_entropy = -torch.log_softmax(torch.cat((similarity[roi], neg_sim)) / temperature, dim=0)[0]

    return torch.mean(cross_entropy)
