import pandas as pd


def csv_loader(
    gt_path="../train_data/train_gt.csv",
    wavelength_path="../train_data/wavelengths.csv",
):
    gt_df = pd.read_csv(gt_path)
    wavelength_df = pd.read_csv(wavelength_path)
    return (gt_df, wavelength_df)
