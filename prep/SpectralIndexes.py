import math
from matplotlib import pyplot as plt
import numpy as np

from prep.SmoothDerivativeBlock import savgol_block


def calc_EVI(input_data: np.ndarray):
    return 2.5 * (
        (input_data[:, :, :, 127] - input_data[:, :, :, 61])
        / (
            input_data[:, :, :, 127]
            + 6 * input_data[:, :, :, 61]
            - 7.5 * input_data[:, :, :, 8]
            + 1
        )
    )


def draw_PVI(input_data: np.ndarray):
    plt.figure(figsize=(30, 30))
    x = input_data[:, :, :, 61].flatten()
    y = input_data[:, :, :, 127].flatten()
    plt.scatter(x, y)
    plt.xticks(np.arange(0, 2000, 50))
    plt.yticks(np.arange(0, 2000, 50))
    plt.ylim((0, 2500))


def calc_PVI(input_data: np.ndarray):
    return (
        input_data[:, :, :, 127] - 1.269 * input_data[:, :, :, 61] + 4.26
    ) / math.sqrt(1 + 1.269**2)


def calc_SDy(input_data: np.ndarray):
    # 560-640 = 32-57
    derived = savgol_block(input_data, deriv=1)
    area_of_interest = derived[:, :, :, 32:58]
    return np.trapz(area_of_interest, dx=3.2, axis=3)


def calc_indexes(input_data: np.ndarray):
    evi = calc_EVI(input_data)
    pvi = calc_PVI(input_data)
    sdy = calc_SDy(input_data)
    return np.stack([evi, pvi, sdy], axis=-1)
