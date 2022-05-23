import numpy as np
from connectome import Transform, optional, impure


class CreateRotationLabel(Transform):
    __inherit__ = True

    @impure
    def label():
        # we can rotate image by {0, 90, 180, 270} in each axis so we have 10 classes
        return np.random.randint(0,10)


class Rotate(Transform):
    __inherit__ = True

    def image(image, /, label):
        if label == 0:
            new_image = image
        elif label == 1:
            new_image = np.transpose(np.flip(image, 1), (1, 0, 2))  
        elif label == 2:
            new_image = np.flip(image, (0, 1))  
        elif label == 3:
            new_image = np.flip(np.transpose(image, (1, 0, 2)), 1)  
        elif label == 4:
            new_image = np.transpose(np.flip(image, 1), (0, 2, 1))  
        elif label == 5:
            new_image = np.flip(image, (1, 2))  
        elif label == 6:
            new_image = np.flip(np.transpose(image, (0, 2, 1)), 1) 
        elif label == 7:
            new_image = np.transpose(np.flip(image, 0), (2, 1, 0)) 
        elif label == 8:
            new_image = np.flip(image, (0, 2)) 
        elif label == 9:
            new_image = np.flip(np.transpose(image, (2, 1, 0)), 0)

        return new_image

    cancer = optional(image)
