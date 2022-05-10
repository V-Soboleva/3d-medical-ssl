import numpy as np


def min_max_scale(x, axis, eps=1e-8):
    x = x - x.min(axis=axis, keepdims=True)
    return x / (x.max(axis=axis, keepdims=True) + eps)


def scale_hu(image_hu, window_hu):
    min_hu, max_hu = window_hu
    assert min_hu < max_hu
    return np.clip((image_hu - min_hu) / (max_hu - min_hu), 0, 1)
