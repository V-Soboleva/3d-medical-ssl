import typing as tp
import numpy as np
from dpipe.im.box import mask2bounding_box as mask_to_box
from pathlib import Path


def normalize_axis(axis, ndim):
    return list(np.core.numeric.normalize_axis_tuple(axis, ndim))
