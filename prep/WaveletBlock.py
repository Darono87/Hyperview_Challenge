import numpy as np
import pywt


def wavelet_transform1d(input_data: np.ndarray, family="haar", result_wave_count=150):
    transformed = np.empty((input_data.shape[0], result_wave_count))
    level = pywt.dwt_max_level(input_data.shape[1], filter_len=family)
    for x in range(input_data.shape[0]):
        transformed[x, :] = np.concatenate(
            pywt.wavedec(input_data[x, :], family, mode="symmetric", level=level)
        )
    return transformed


def wavelet_transform(input_data: np.ndarray, family="haar", result_wave_count=150):
    transformed = np.empty(
        (
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            result_wave_count,
        )
    )
    level = pywt.dwt_max_level(input_data.shape[3], filter_len=family)

    for x in range(input_data.shape[0]):
        for y in range(input_data.shape[1]):
            for z in range(input_data.shape[2]):
                transformed[x, y, z, :] = np.array(
                    np.concatenate(
                        (
                            pywt.wavedec(
                                input_data[x, y, z, :],
                                family,
                                mode="symmetric",
                                level=level,
                            )
                        )
                    )
                )

    return transformed
