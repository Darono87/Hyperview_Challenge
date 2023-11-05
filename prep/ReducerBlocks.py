import math
import numpy as np


def mean_1d_block(data):
    new_data = np.empty(shape=(len(data), data[0].shape[0]))

    for index in range(len(new_data)):
        new_data[index] = data[index].mean((1, 2))

    return new_data


def median_1d_block(data):
    new_data = np.empty(shape=(len(data), data[0].shape[0]))

    for index in range(len(new_data)):
        new_data[index] = np.ma.median(data[index], axis=(1, 2))

    return new_data


def quantiles_block(data):
    new_data = np.empty(shape=(len(data), 11, data[0].shape[0]))
    for index in range(len(new_data)):
        q1 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0, axis=(1, 2)
        )
        q2 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.1, axis=(1, 2)
        )
        q3 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.2, axis=(1, 2)
        )
        q4 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.3, axis=(1, 2)
        )
        q5 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.4, axis=(1, 2)
        )
        q6 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.5, axis=(1, 2)
        )
        q7 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.6, axis=(1, 2)
        )
        q8 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.7, axis=(1, 2)
        )
        q9 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.8, axis=(1, 2)
        )
        q10 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 0.9, axis=(1, 2)
        )
        q11 = np.nanquantile(
            data[index].astype("float32").filled(np.nan), 1, axis=(1, 2)
        )
        new_data[index] = np.array([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11])

    return new_data.transpose((0, 2, 1))
