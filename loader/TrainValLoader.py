import numpy as np


def one_train_val_loader(
    number: int, get_hsi_data_path=lambda num: f"../train_data/train_data/{num}.npz"
):
    with np.load(get_hsi_data_path(number)) as npz:
        arr = np.ma.MaskedArray(**npz)
    return arr


def train_val_loader(
    get_hsi_data_path=lambda num: f"../train_data/train_data/{num}.npz",
):
    all_data_spectral = []

    for i in range(0, 1732):
        with np.load(get_hsi_data_path(i)) as npz:
            data = np.ma.MaskedArray(**npz)
        all_data_spectral.append(data)

    return all_data_spectral
