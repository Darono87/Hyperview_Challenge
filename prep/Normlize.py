import numpy as np


def normalize_layers(input_data: np.ndarray, should_calc_ref: bool):
    if should_calc_ref:
        normalize_layers.mean = input_data.mean(axis=(0))
        normalize_layers.std = input_data.std(axis=(0))

    input_data = (input_data - normalize_layers.mean) / normalize_layers.std

    return np.ma.masked_array(input_data, mask=np.isnan(input_data)).filled(0)
