import numpy as np


def one_test_loader(
    number: int, get_hsi_test_path=lambda num: f"../test_data/{num}.npz"
):
    with np.load(get_hsi_test_path(number)) as npz:
        arr = np.ma.MaskedArray(**npz)
    return arr


def test_loader(get_hsi_test_path=lambda num: f"../test_data/{num}.npz"):
    all_data_spectral = []

    for i in range(0, 1154):
        with np.load(get_hsi_test_path(i)) as npz:
            data = np.ma.MaskedArray(**npz)
        all_data_spectral.append(data)

    return all_data_spectral
