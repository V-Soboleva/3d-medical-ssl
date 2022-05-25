import typing as tp
import numpy as np
import random
import torch
from torchio import RandomElasticDeformation
from dpipe.im.box import mask2bounding_box as mask_to_box


def random_elasticdeform(x, seed=0, num_control_points=10, max_displacement=(10, 10, 0), **kwargs):
    transform = RandomElasticDeformation(
        num_control_points=num_control_points,
        max_displacement=max_displacement,
        locked_borders=2,  # displacement of control points at the border of the image will also be set to 0.
        image_interpolation='nearest',
        **kwargs
    )
    torch.manual_seed(seed)
    return transform(x)


class Transform2D(tp.NamedTuple):
    dims: tp.Tuple[int, int]
    vflip: bool
    hflip: bool
    angle: float
    scale: float
    vshift: float
    hshift: float
    elastic: bool
    elasticdeform_seed: int

    @classmethod
    def random(cls, dims: tp.Tuple[int, int], vflip_p: float, hflip_p: float,
               max_angle: float, max_scale: float, max_shift: float, elastic_p : float):
        vflip = random.random() < vflip_p
        hflip = random.random() < hflip_p
        angle = random.uniform(-max_angle, max_angle)
        scale = random.uniform(1, max_scale)
        vshift = random.random() * max_shift
        hshift = random.random() * max_shift
        elastic = random.random() < elastic_p
        elasticdeform_seed = np.random.randint(2147483647)

        return cls(dims, vflip, hflip, angle, scale, vshift, hshift, elastic, elasticdeform_seed)

    def __call__(
            self,
            x: torch.Tensor,
            fill_value: tp.Optional[float] = None,
            dims: tp.Optional[tp.Tuple[int, int]] = None,
            **kwargs
    ) -> torch.Tensor:
        """Apply 2D transformation in ``dims`` plane.

        Args:
            x (torch.Tensor): input tensor with ndim >= 2.
            fill_value (float, optional): Value to fill empty regions. Defaults to 0.0.

        Returns:
            torch.Tensor: tensor of the same shape as input.
        """
        from torchvision.transforms.functional import hflip, vflip, affine

        if dims is None:
            dims = self.dims

        x = torch.movedim(x, dims, (-2, -1))

        ndim = x.ndim
        assert ndim >= 2
        if ndim == 2:
            x = x.unsqueeze(0)

        if self.hflip:
            x = hflip(x)
        if self.vflip:
            x = vflip(x)

        translate = (self.hshift * x.shape[-1] * self.scale,
                     self.vshift * x.shape[-2] * self.scale)
        x = affine(x, angle=self.angle, translate=translate,
                   scale=self.scale, shear=0, fill=fill_value)

        if ndim == 2:
            x = x.squeeze(0)

        x = torch.movedim(x, (-2, -1), dims)

        device = x.device
        if self.elastic:
            x = random_elasticdeform(x.cpu().detach(), seed=self.elasticdeform_seed, **kwargs).to(device)

        return x
