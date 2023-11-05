from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity_based import reliefF


def reliefF_method(data, param_to_compare, final_count=60):
    averaged_data = data.mean(axis=(1, 2))
    select = SelectKBest(score_func=reliefF.reliefF, k=final_count)
    select.fit(averaged_data, param_to_compare)
    indices = select.get_support(True)

    return data[:, :, :, indices], lambda d: d[:, :, :, indices], indices
