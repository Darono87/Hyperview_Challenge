import math
import numpy as np
from skimage.transform import resize
from skimage.restoration import inpaint_biharmonic


def make_resize_block(
    output_length, pre_fill=None, order=1, extend_strategy="resize", post_fill=None
):
    def resize_block(data: np.ndarray):
        new_shape = (
            len(data),
            output_length,
            output_length,
            data[0].shape[0],
        )
        new_data = np.ma.array(
            data=np.empty(shape=new_shape),
            mask=np.full(
                new_shape,
                False,
            ),
        )

        for index, photo in enumerate(data):
            transposed = np.transpose(photo, (1, 2, 0))
            filled = (
                inpaint_biharmonic(
                    transposed.data, transposed.mask[:, :, 0], channel_axis=2
                )
                if pre_fill == "harmonic"
                else np.ma.filled(transposed, pre_fill)
                if pre_fill != None
                else transposed
            )
            resized_photo_size = (
                output_length,
                output_length,
                data[0].shape[0],
            )
            if (
                filled.shape[0] < output_length
                and filled.shape[1] < output_length
                and extend_strategy == "padding"
            ):
                pad_widths = (
                    (
                        math.floor((output_length - filled.shape[0]) / 2),
                        math.ceil((output_length - filled.shape[0]) / 2),
                    ),
                    (
                        math.floor((output_length - filled.shape[1]) / 2),
                        math.ceil((output_length - filled.shape[1]) / 2),
                    ),
                    (0, 0),
                )

                new_data[index] = np.pad(
                    filled,
                    pad_widths,
                    mode="constant",
                    constant_values=pre_fill,
                )
                new_data[index][
                    pad_widths[0][0] : -pad_widths[0][1],
                    pad_widths[1][0] : -pad_widths[1][1],
                    :,
                ] = filled
                new_data[index].mask = np.pad(
                    transposed.mask,
                    pad_widths,
                    mode="constant",
                    constant_values=True,
                )
                new_data[index].mask[
                    pad_widths[0][0] : -pad_widths[0][1],
                    pad_widths[1][0] : -pad_widths[1][1],
                    :,
                ] = transposed.mask

                continue

            resized_photo = resize(
                filled,
                resized_photo_size,
                anti_aliasing=True,
                order=order,
                preserve_range=True,
            )
            resized_mask = resize(
                transposed.mask,
                resized_photo_size,
                mode="reflect",
                order=0,
            )
            new_data[index] = (
                np.ma.filled(resized_photo, post_fill)
                if post_fill != None
                else resized_photo
            )
            new_data[index].mask = resized_mask

        return new_data

    return resize_block
