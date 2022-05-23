import typing as tp
import numpy as np


def normalize_axis(axis, ndim):
    return list(np.core.numeric.normalize_axis_tuple(axis, ndim))


def generate_permutation(patches_per_side = 3, num_permutations = 100):
    permutations = []
    permutation_len = patches_per_side ** 3
    counter = 0
    flag = True
    while counter < num_permutations:
            pp = np.random.permutation(permutation_len)
            for perm in permutations:
                if (perm == pp).all():
                    flag = False
                    break
            if flag == True:
                counter += 1
                permutations.append(pp)

    permutations = np.stack(permutations)
    return permutations

