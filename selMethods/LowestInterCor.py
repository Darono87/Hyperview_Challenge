import numpy as np


def lowest_inter_cor_method(data_cube, variance_filter=100, final_count=60):
    variances = data_cube.mean(axis=(1, 2)).var(axis=(0))
    most_variance_indices = np.argsort(variances)[::-1]
    most_variance = data_cube[:, :, :, most_variance_indices[:variance_filter]]
    cors = np.corrcoef(
        most_variance.reshape(
            most_variance.shape[0] * most_variance.shape[1] * most_variance.shape[2],
            most_variance.shape[3],
        ).transpose((1, 0))
    )
    least_correlated_indices = [0]

    for i in range(final_count - 1):
        start_min = 1.01
        new_element = None
        for_collection = np.abs(cors)
        for x in range(cors.shape[0]):
            max_col = 0
            if x in least_correlated_indices:
                continue
            for y in least_correlated_indices:
                max_col = max(max_col, for_collection[x, y])
            new_element = x if max_col < start_min else new_element
            start_min = min(start_min, max_col)
        least_correlated_indices.append(new_element)
    least_correlated_indices.sort()
    most_relevant_bands = most_variance[:, :, :, least_correlated_indices]
    return (
        most_relevant_bands,
        lambda data: (data[:, :, :, most_variance_indices[:variance_filter]])[
            :, :, :, least_correlated_indices
        ],
    )
