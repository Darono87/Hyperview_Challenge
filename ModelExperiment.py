import gc
import math
from typing import List, Type
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class ModelExperiment:
    def __init__(
        self,
    ):
        self.models = []
        self.train_history = []
        self.val_history = []
        self.partial_scores = []

    def get_errors_chart(self):
        plt.figure(figsize=(20, 6))
        for index, model in enumerate(self.models):
            remainder = len(model.samplewise_real) % 12
            averaged = np.reshape(
                (model.calc_error() / model.samplewise_real)[:-remainder],
                (math.floor(len(model.samplewise_real) / 12), 12),
            ).mean(axis=1)
            plt.plot(
                averaged
                + [(model.calc_error() / model.samplewise_real)[-remainder:].mean()],
                label=f"Model {index}",
            )
        plt.title(f"Średni relatywny błąd w grupach po 12 próbek")
        plt.xlabel("Próbka")
        plt.ylabel("AE dla próbki")
        plt.legend()

    def get_errors_report(self, path="./experiments/result.csv"):
        if len(self.models) == 0:
            return "No models to generate report from."
        errors_df = {
            "Index": np.arange(1, len(self.models[0].samplewise_real) + 1),
            "Ground Truth": self.models[0].samplewise_real,
        }
        for index, model in enumerate(self.models):
            errors_df = {
                **errors_df,
                f"Predicted {index}": model.samplewise_predicted,
                f"MAE {index}": model.calc_error(),
            }
        errors_df = pd.DataFrame(errors_df)
        return errors_df.to_csv(path)

    def sketch_results(self, ylim, ylim_val):
        if len(self.models) == 0:
            return
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.grid(True)

        history = []

        for i, model in enumerate(self.models):
            if not hasattr(model, "read_history"):
                continue
            mean_train_mae, mean_val_mae, mean_partial_score, _ = model.read_history()
            epochs = range(1, len(mean_train_mae) + 1)
            history.append(
                {
                    "epochs": epochs,
                    "mean_train_mae": mean_train_mae,
                    "mean_val_mae": mean_val_mae,
                    "mean_partial_score": mean_partial_score,
                }
            )

        for model in history:
            plt.plot(model["epochs"], model["mean_train_mae"], label=i)

        plt.title(f"MAE Treningowe")
        plt.xlabel("Epoki")
        plt.ylabel("MAE")
        plt.ylim(ylim)
        if len(self.train_history) > 1:
            plt.legend()

        # -------------------

        plt.subplot(1, 3, 2)
        plt.grid(True)
        for model in history:
            print(model["mean_val_mae"])
            plt.plot(model["epochs"], model["mean_val_mae"], label=i)

        plt.title(f"MAE Walidacyjne")
        plt.xlabel("Epoki")
        plt.ylabel("MAE")
        plt.ylim(ylim_val)
        if len(self.val_history) > 1:
            plt.legend()

        # -------------------

        plt.subplot(1, 3, 3)
        plt.grid(True)
        for model in history:
            plt.plot(model["epochs"], model["mean_partial_score"], label=i)

        plt.title(f"Wynik cząstkowy")
        plt.xlabel("Epoki")
        plt.ylabel("Wynik cząstkowy")
        plt.ylim((0.5, 1.5))
        if len(self.partial_scores) > 1:
            plt.legend()

    def run_experiment(
        self,
        params_vector: List,
        data: [np.ndarray],
        gt: [np.ndarray],
        test_data=[np.ndarray],
        folds=5,
    ):
        self.models = []
        self.train_history = []
        self.val_history = []
        self.partial_scores = []
        for index, params in enumerate(params_vector):
            gc.collect()
            if getattr(params, "generate_flow", False):
                flow = params.generate_flow(index)
            else:
                flow = params.generic.generate_flow(index)

            if getattr(params, "generic", False):
                prepare_function = params.generic.prepare_function
            else:
                prepare_function = params.prepare_function

            flow.cross_val(
                data[index], gt[index], prepare_function, test_data[index], folds
            )
            self.models.append(flow)

        return self.models
