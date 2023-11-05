import numpy as np
import scipy.signal, scipy.interpolate


def continuum_removal(input_data: np.ndarray):
    transformed = np.empty_like(input_data)
    for x in range(input_data.shape[0]):
        for y in range(input_data.shape[1]):
            for z in range(input_data.shape[2]):
                peaks, _ = scipy.signal.find_peaks(input_data[x, y, z, :])
                indices = np.concatenate(([0], peaks, [input_data.shape[3] - 1]))
                continuum = np.interp(
                    range(input_data.shape[3]),
                    indices,
                    input_data[x, y, z, indices],
                )
                transformed[x, y, z, :] = input_data[x, y, z, :] / continuum

    return transformed


def continuum_removal1d(input_data: np.ndarray):
    transformed = np.empty_like(input_data)
    for x in range(input_data.shape[0]):
        peaks, _ = scipy.signal.find_peaks(input_data[x, :])
        indices = np.concatenate(([0], peaks, [input_data.shape[1] - 1]))
        continuum = np.interp(
            range(input_data.shape[1]),
            indices,
            input_data[x, indices],
        )
        transformed[x, :] = input_data[x, :] / continuum

    return transformed
