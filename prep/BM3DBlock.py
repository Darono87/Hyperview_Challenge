import math
import numpy as np
from bm3d import bm3d


def bm3d_block(input_data: np.ndarray, psd=1, final_layers=5):
    transformed_cube = np.mean(
        input_data.reshape(
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            final_layers,
            math.floor(input_data.shape[3] / final_layers),
        ),
        axis=4,
    )
    result_images = []
    for image in range(transformed_cube.shape[0]):
        result_images.append(bm3d(transformed_cube[image, :, :, :], psd))
    return np.ma.array(result_images, mask=input_data.mask[:, :, :, :final_layers])
