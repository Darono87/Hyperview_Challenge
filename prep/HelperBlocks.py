from functools import reduce

import numpy as np


def split_3d_to_timeseries(data, time_length=15):
    transposed = np.transpose(data, (0, 3, 1, 2))
    return transposed.reshape(
        (
            transposed.shape[0],
            time_length,
            round(transposed.shape[1] / time_length),
            transposed.shape[2],
            transposed.shape[3],
            1,
        ),
    )


def pipe(data, functions):
    return reduce(
        lambda acc, func: func(acc),
        functions,
        data,
    )
