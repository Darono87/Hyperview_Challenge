from typing import Callable, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import sklearn as sk
import skops.io as sio
from flow.ModelFlow import ModelFlow, calc_partial_score


class ClassicalModelFlow(ModelFlow):
    def __init__(
        self,
        get_model: Callable[[np.ndarray], tf.keras.models.Model] = None,
    ):
        self.model = None
        self.base_get_model = get_model

    def get_model(self, train_labels):
        return self.base_get_model(train_labels)

    def export_model(self, pathname="./experiments/exported_model"):
        binary_data = sio.dumps(self.model)
        with open(pathname, "wb") as file:
            file.write(binary_data)

    def evaluate(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        train_labels: Union[np.ndarray, None],
    ):
        predicted = self.predict(val_data)
        score = calc_partial_score(train_labels, predicted, val_labels)
        r2 = r2_score(val_labels, predicted)
        loss = mae = mean_absolute_error(val_labels, predicted)
        return loss, score, mae, r2

    def predict(self, data: np.ndarray):
        return self.model.predict(data)

    def inner_fit(
        self,
        train: np.ndarray,
        _: np.ndarray,
        train_gt: np.ndarray,
        __: np.ndarray,
    ):
        self.model.fit(train, train_gt)
