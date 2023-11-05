import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def mutual_info_method(data_cube, param_to_compare, final_count=60):
    layers_mi = []
    for layer in range(data_cube.shape[3]):
        layers_mi.append(
            mutual_info_regression(
                np.expand_dims(
                    data_cube[:, :, :, layer].mean(axis=(1, 2)),
                    -1,
                ),
                param_to_compare,
            )
        )
    layers_mi = np.array(layers_mi).flatten()
    max_mi = (np.argsort(layers_mi)[::-1])[:final_count]
    return data_cube[:, :, :, max_mi], lambda d: d[:, :, :, max_mi], max_mi
