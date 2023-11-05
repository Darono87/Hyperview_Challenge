import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def gaussian_block1d(input_data: np.ndarray, radius=5, sigma=1):
    return gaussian_filter1d(
        input_data,
        axis=1,
        radius=radius,
        sigma=sigma,
    )


def gaussian_block(input_data: np.ndarray, radius=1, sigma=0.7):
    return np.ma.array(
        gaussian_filter(
            input_data,
            axes=(1, 2, 3),
            radius=radius,
            sigma=sigma,
        ),
        mask=input_data.mask,
    )
