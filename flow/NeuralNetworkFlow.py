import gc
from typing import Callable, List, Union
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
from flow.ModelFlow import ModelFlow


class NeuralNetworkFlow(ModelFlow):
    def __init__(
        self,
        get_model: Callable[[np.ndarray], tf.keras.models.Model] = None,
        get_callbacks: Callable[..., List[tf.keras.callbacks.Callback]] = lambda: [
            tf.keras.callbacks.History(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.6, patience=10, min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="experiment1/checkpoints/checkpoint_{epoch:02d}-{val_score_metric:.2f}.h5",
                monitor="val_score_metric",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                save_freq="epoch",
            ),
        ],
        epochs=30,
    ):
        self.model: tf.keras.Model = None
        self.get_callbacks = get_callbacks
        self.base_get_model = get_model
        self.epochs = epochs

    def get_model(self, train_labels):
        gc.collect()
        model = self.base_get_model(train_labels)
        return model

    def read_history(self):
        train_mae = np.array(
            list(map(lambda h: h.history["mean_absolute_error"], self.histories))
        )
        partial_score = np.array(
            list(map(lambda h: h.history["val_score_metric"], self.histories))
        )
        val_mae = np.array(
            list(map(lambda h: h.history["val_mean_absolute_error"], self.histories))
        )
        lr = np.array(list(map(lambda h: h.history["lr"], self.histories)))
        mean_train_mae = train_mae.mean(0)
        mean_val_mae = val_mae.mean(0)
        mean_partial_score = partial_score.mean(0)
        mean_lr = lr.mean(0)
        return (mean_train_mae, mean_val_mae, mean_partial_score, mean_lr)

    def save_history(self, filename="./experiments/train_report.csv"):
        train, val, scores, lr = self.read_history()
        errors_df = pd.DataFrame(
            {
                "Index": np.arange(1, len(train) + 1),
                "Dokładność tr.": train,
                "Dokładność wal.": val,
                "Ocena cząstkowa": scores,
                "Współczynnik uczenia": lr,
            }
        )
        errors_df.to_csv(filename)

    def predict(self, data: np.ndarray):
        if self.model == None:
            return
        predicted = self.model.predict(data)
        return predicted.reshape(-1)

    def inner_fit(
        self,
        train: np.ndarray,
        val: np.ndarray,
        train_gt: np.ndarray,
        val_gt: np.ndarray,
    ):
        history = self.model.fit(
            train,
            train_gt,
            epochs=self.epochs,
            validation_data=(val, val_gt),
            callbacks=self.get_callbacks(),
        )
        self.histories.append(history)

    def export_model(self, pathname="./experiments/exported_model"):
        if self.model != None:
            self.model.save(pathname, save_format="h5")

    def export_weights(self, pathname="./experiments/exported_weights"):
        if self.model != None:
            self.model.save_weights(pathname)

    def import_weights(self, input_shape, pathname="./experiments/exported_weights"):
        if self.model == None:
            self.model = self.get_model([])
            self.model.build(input_shape=input_shape)
        self.model.load_weights(pathname)

    def evaluate(
        self, val_data: np.ndarray, val_labels: np.ndarray, _: Union[np.ndarray, None]
    ):
        _, loss, score, mae = self.model.evaluate(val_data, val_labels)
        r2 = r2_score(val_labels, val_labels)
        return loss, score, mae, r2
