import numpy as np
import hyperspy.signals as signal


def pca_block(input_data: np.ndarray, comp):
    transformed = []
    for picture in input_data:
        sig = signal.Signal1D(picture)
        sig.decomposition(
            algorithm="sklearn_pca",
            output_dimension=3,
            print_info=False,
            navigation_mask=picture.mask.transpose((2, 1, 0))[0, :, :],
        )
        # sig.plot_explained_variance_ratio()
        transformed.append(sig.get_decomposition_model(comp).data)

    return np.array(transformed)
