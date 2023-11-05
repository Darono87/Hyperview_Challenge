import numpy as np
from scipy.stats import spearmanr, pearsonr


def spearman_method(data_cube, param_to_compare, final_count=60):
    coefs = []
    for i in range(data_cube.shape[3]):
        layer_data = data_cube[:, :, :, i].mean(axis=(1, 2))
        coef, p_val = spearmanr(layer_data, param_to_compare)
        coefs.append(coef)
    coefs = np.abs(np.array(coefs))
    max_coefs = np.sort((np.argsort(coefs))[::-1][:final_count])
    return (
        data_cube[:, :, :, max_coefs],
        lambda data: data[:, :, :, max_coefs],
        max_coefs,
    )
