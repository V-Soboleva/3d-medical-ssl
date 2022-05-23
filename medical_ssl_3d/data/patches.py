import numpy as np


def get_random_patches(image, patch_size, patches_per_side): 
    h, w, d = image.shape
    patch_size = np.array(patch_size)
    h_patch, w_patch, d_patch = patch_size
    
    assert (patch_size * patches_per_side < image.shape).all()

    h_grid = h // patches_per_side
    w_grid = w // patches_per_side
    d_grid = d // patches_per_side

    patches = []
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            for k in range(patches_per_side):

                p = image[i * h_grid: (i + 1) * h_grid,
                          j * w_grid: (j + 1) * w_grid, 
                          k * d_grid: (k + 1) * d_grid]

                if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                    p = random_crop(p, [h_patch, w_patch, d_patch])

                patches.append(p)

    return np.array(patches)


def random_crop(image, crop_size):
    h, w, d = crop_size[0], crop_size[1], crop_size[2]
    h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]
    d = min(d, d_old)

    # random crop
    x = np.random.randint(0, 1 + h_old - h)
    y = np.random.randint(0, 1 + w_old - w)
    z = np.random.randint(0, 1 + d_old - d)


    return image[x:x + h, y:y + w, z:z + d]


def get_image_from_patches(patches, patches_per_side):
    _, h_patch, w_patch, d_patch = patches.shape
    new_image = np.empty((h_patch * patches_per_side, w_patch * patches_per_side, d_patch * patches_per_side))
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            for k in range(patches_per_side):
                new_image[i * h_patch: (i + 1) * h_patch,
                          j * w_patch: (j + 1) * w_patch, 
                          k * d_patch: (k + 1) * d_patch] = patches[i * patches_per_side ** 2 + \
                                                                    j * patches_per_side ** 1 + \
                                                                    k * patches_per_side ** 0]
    return new_image


def get_random_point(image, crop_size):
    h, w, d = crop_size[0], crop_size[1], crop_size[2]
    h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]
    d = min(d, d_old)

    # random crop
    x = np.random.randint(0, 1 + h_old - h)
    y = np.random.randint(0, 1 + w_old - w)
    z = np.random.randint(0, 1 + d_old - d)


    return [x, y, z]
