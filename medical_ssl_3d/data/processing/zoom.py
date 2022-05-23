import typing as tp
import numpy as np

from connectome import Transform, Mixin, optional
from dpipe.im import zoom

from ..utils import normalize_axis


class _Zoom(Mixin):
    def image(image, _scale_factor, _axis):
        return zoom(image, _scale_factor, _axis)

    def spacing_mm(spacing_mm, _scale_factor):
        return tuple(np.array(spacing_mm) / _scale_factor)

    @optional
    def mask(mask, _scale_factor, _axis):
        return zoom(mask, _scale_factor, _axis, order=0)


class Zoom(Transform, _Zoom):
    _axis: tp.Union[int, tp.Sequence[int]]

    def _scale_factor(scale_factor):
        return scale_factor


class ZoomToSpacing(Transform, _Zoom):
    _to_spacing: tp.Sequence[float]
    _axis: tp.Union[int, tp.Sequence[int]]

    def _scale_factor(spacing_mm, _to_spacing):
        to_spacing = np.array(_to_spacing, dtype=float)
        return np.nan_to_num(spacing_mm / to_spacing, nan=1)


class ZoomToShape(Transform, _Zoom):
    _shape: tp.Union[int, tp.Sequence[int]]
    _axis: tp.Union[int, tp.Sequence[int]]

    def _scale_factor(image, _shape, _axis):
        axis = normalize_axis(_axis, image.ndim)
        old_shape = np.array(image.shape, dtype=float)
        return _shape / old_shape[axis]
