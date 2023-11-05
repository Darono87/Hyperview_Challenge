from sklearn.feature_selection import SelectKBest
from skfeature.function.information_theoretical_based import FCBF


def fcbf_method(data, param_to_compare, final_count=60):
    averaged_data = data.mean(axis=(1, 2))
    select = SelectKBest(score_func=FCBF.fcbf, k=final_count)
    select.fit(averaged_data, param_to_compare)
    indices = select.get_support(True)

    return data[:, :, :, indices], lambda d: d[:, :, :, indices], indices
