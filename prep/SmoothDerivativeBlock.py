import numpy as np
from scipy.signal import savgol_filter


def savgol_block1d(
    input_data: np.ndarray, window_size=5, polyorder=2, deriv=0, delta=1
):
    return savgol_filter(
        input_data,
        window_length=window_size,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=1,
    )


def savgol_block(input_data: np.ndarray, window_size=5, polyorder=2, deriv=0, delta=1):
    return np.ma.array(
        savgol_filter(
            input_data,
            window_length=window_size,
            polyorder=polyorder,
            deriv=deriv,
            axis=3,
            delta=delta,
        ),
        mask=input_data.mask,
    )
