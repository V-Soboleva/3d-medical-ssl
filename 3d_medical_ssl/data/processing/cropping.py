import typing as tp
import numpy as np

from connectome import Transform, Mixin, impure, positional
from dpipe.im.box import returns_box
from dpipe.im import crop_to_box
from dpipe.im.shape_ops import crop_to_shape, pad_to_shape
from dpipe.im.patch import get_random_box

from ..utils import normalize_axis


class _CropToBox(Mixin):
    @positional
    def image(x, _box, _axis):
        return crop_to_box(x, _box, _axis)

    mask = image


class CropToBox(Transform, _CropToBox):
    __inherit__ = 'spacing_mm'
    _axis: tp.Union[tp.Sequence[int], int]

    def _box(cropping_box):
        return cropping_box


class GetRandomPatch(Transform, _CropToBox):
    __inherit__ = 'spacing_mm'
    _patch_size: int
    _axis: tp.Union[tp.Sequence[int], int]

    @impure
    @returns_box
    def _box(shape, _patch_size, _axis):
        axis = normalize_axis(_axis, len(shape))
        shape = np.array(shape)
        return get_random_box(shape[axis], _patch_size)


class CropOrPad(Transform):
    __inherit__ = 'spacing_mm'
    _shape: tp.Union[tp.Sequence[int], int]
    _axis: tp.Union[tp.Sequence[int], int]

    def image(image, /, _shape, _axis):
        axis = normalize_axis(_axis, image.ndim)
        if np.all(np.array(image.shape)[axis] >= np.array(_shape)):
            return crop_to_shape(image, _shape, _axis)
        else:
            return pad_to_shape(image, _shape, _axis, 0)

    mask = image
    

