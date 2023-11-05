import numpy as np
from prep.ReducerBlocks import mean_1d_block


def msc_block1d(input_data: np.ndarray, should_calc_ref: bool, fill=0, rank=1):
    data_msc = np.full_like(input_data, 0)

    if should_calc_ref:
        msc_block.ref = input_data.mean((0))
    for sample_i in range(input_data.shape[0]):
        if np.ma.is_masked(input_data[sample_i, :]):
            data_msc[sample_i, :] = fill
            continue

        fit = np.ma.polyfit(msc_block.ref, input_data[sample_i, :], rank, full=True)

        if rank == 1:
            data_msc[sample_i, :] = (
                (input_data[sample_i, :] - fit[0][1]) / fit[0][0]
                if fit[0][0] != 0
                else fit[0][1]
            )
        else:
            data_msc[sample_i, :] = (
                -fit[0][1]
                + np.ma.sqrt(
                    fit[0][1] ** 2
                    - 4 * fit[0][0] * (fit[0][2] - input_data[sample_i, :])
                )
            ) / (2 * fit[0][0])

    return data_msc


def msc_block(input_data: np.ndarray, should_calc_ref: bool, fill=0, rank=1):
    data_msc = np.full_like(input_data, fill)

    if should_calc_ref:
        msc_block.ref = input_data.mean((0, 1, 2))

    for sample_i in range(input_data.shape[0]):
        for y in range(input_data.shape[1]):
            for x in range(input_data.shape[2]):
                if np.ma.is_masked(input_data[sample_i, y, x, :]):
                    data_msc[sample_i, y, x, :] = fill
                    continue
                fit = np.polyfit(
                    msc_block.ref, input_data[sample_i, y, x, :], rank, full=True
                )

                if rank == 1:
                    data_msc[sample_i, y, x, :] = (
                        (input_data[sample_i, y, x, :] - fit[0][1]) / fit[0][0]
                        if fit[0][0] != 0
                        else fit[0][1]
                    )
                else:
                    data_msc[sample_i, y, x, :] = (
                        -fit[0][1]
                        + np.ma.sqrt(
                            fit[0][1] ** 2
                            - 4
                            * fit[0][0]
                            * (fit[0][2] - input_data[sample_i, y, x, :])
                        )
                    ) / (2 * fit[0][0])

    return (
        np.ma.array(data_msc, mask=input_data.mask)
        if hasattr(input_data, "mask")
        else data_msc
    )
