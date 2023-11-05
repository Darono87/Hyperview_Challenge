from typing import List
import pandas as pd
import numpy as np
from ModelFlow import ModelFlow


class ModelCollection:
    def __init__(
        self,
        k_model: ModelFlow,
        p_model: ModelFlow,
        pH_model: ModelFlow,
        mg_model: ModelFlow,
    ):
        self.models = {"K": k_model, "P": p_model, "pH": pH_model, "Mg": mg_model}

    def predict(self, image_sets: List[np.ndarray]):
        results = pd.DataFrame(columns=["P", "K", "Mg", "pH"])
        for key, images in zip(self.models.keys(), image_sets):
            results[key] = self.models[key].predict(images)
        return results

    def get_errors_report(self, path="./experiments/result.csv"):
        errors_df = pd.DataFrame(
            {
                "Index": np.arange(1, len(self.models["K"].samplewise_predicted) + 1),
                "Predicted K": self.models["K"].samplewise_predicted,
                "Ground Truth K": self.models["K"].samplewise_real,
                "MAE K": self.models["K"].calc_error(),
                "Predicted P": self.models["P"].samplewise_predicted,
                "Ground Truth P": self.models["P"].samplewise_real,
                "MAE P": self.models["P"].calc_error(),
                "Predicted pH": self.models["pH"].samplewise_predicted,
                "Ground Truth pH": self.models["pH"].samplewise_real,
                "MAE pH": self.models["pH"].calc_error(),
                "Predicted Mg": self.models["Mg"].samplewise_predicted,
                "Ground Truth Mg": self.models["Mg"].samplewise_real,
                "MAE Mg": self.models["Mg"].calc_error(),
            }
        )
        errors_df.to_csv(path)

    def fit(
        self,
        train: np.ndarray,
        val: np.ndarray,
        train_gt: np.ndarray,
        val_gt: np.ndarray,
    ):
        elapsed_times = {}
        for key, modelContainer in self.models.items():
            print(f"Work for {key}:")
            elapsed_times[key] = modelContainer.fit(
                train,
                val,
                train_gt[key].values,
                val_gt[key].values,
            )
        return elapsed_times

    def calc_final_score(
        self, train_gt: np.ndarray, val: np.ndarray, val_gt: pd.DataFrame
    ):
        scores = []
        for key, model in self.models.items():
            score = model.calc_partial_score(
                train_gt[key].values, val, val_gt[key].values
            )
            scores.append(score)
        return (scores[0], scores[1], scores[2], scores[3])

    def make_submission(self, test_data: List[np.ndarray], path="./submission.csv"):
        all_test = self.predict(test_data)
        all_test["sample_index"] = range(0, 1154)
        all_test.set_index("sample_index").to_csv(path)
