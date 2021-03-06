import numpy as np
import typing as tp
from connectome import Transform, optional, impure
from dpipe.io import load

from medical_ssl_3d.data.patches import get_random_patches, get_image_from_patches


class CreateJigLabel(Transform):
    __inherit__ = True
    _num_permutations: int = 100

    @impure
    def label(_num_permutations):
        # along all possible permiutations we leave num_permutations
        return np.random.randint(0, _num_permutations)


class Permute(Transform):
    __inherit__ = True
    _jitter: int
    _patches_per_side: int = 3
    _permutations_path: str = 'permutations.json'

    @impure
    def image(image, /, label, _permutations_path, _patches_per_side, _jitter):
        im = image[0].copy()
        permutations = load(_permutations_path)
        patches = get_random_patches(im, _jitter, _patches_per_side)
        permut_patches = patches[np.array(permutations[label])]
        new_image = get_image_from_patches(permut_patches, _patches_per_side)

        return np.expand_dims(new_image, axis=0)

    cancer = optional(image)
