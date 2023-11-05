from abc import ABC, abstractmethod
import gc
from typing import Union
import time
import pandas as pd
import numpy as np
import tensorflow as tf


def calc_partial_score(
    train_gt: np.ndarray,
    predicted_gt: np.ndarray,
    val_gt: np.ndarray,
):
    mean = np.mean(train_gt, axis=0)
    baseline_prediciton = np.full(len(val_gt), mean)
    baseline_mse = np.mean((val_gt - baseline_prediciton) ** 2, axis=0)
    our_mse = np.mean((val_gt - predicted_gt) ** 2, axis=0)
    return our_mse / baseline_mse


class ModelFlow(ABC):
    @abstractmethod
    def get_model(self, train_labels):
        pass

    @abstractmethod
    def export_model(self, pathname="./experiments/exported_model"):
        pass

    @abstractmethod
    def evaluate(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        train_labels: Union[np.ndarray, None],
    ):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray):
        pass

    @abstractmethod
    def inner_fit(
        self,
        train: np.ndarray,
        val: np.ndarray,
        train_gt: np.ndarray,
        val_gt: np.ndarray,
    ):
        pass

    def fit(
        self,
        train: np.ndarray,
        val: np.ndarray,
        train_gt: np.ndarray,
        val_gt: np.ndarray,
        continue_training=False,
    ):
        if self.model == None and self.base_get_model == None:
            return
        if self.model == None or not continue_training:
            self.model = self.get_model(train_gt)
        if not continue_training:
            self.reset_performance()
        start_time = time.time()
        self.inner_fit(train, val, train_gt, val_gt)
        elapsed_time = time.time() - start_time
        return elapsed_time

    def calc_error(self):
        return np.abs(
            np.array(self.samplewise_real) - np.array(self.samplewise_predicted)
        )

    def calc_partial_score(
        self,
        train_gt: np.ndarray,
        val: np.ndarray,
        val_gt: np.ndarray,
    ):
        our_results = self.predict(val)
        return calc_partial_score(train_gt, our_results, val_gt)

    def get_errors_report(self, path="./experiments/result.csv"):
        errors_df = pd.DataFrame(
            {
                "Index": np.arange(1, len(self.samplewise_predicted) + 1),
                "Predicted": self.samplewise_predicted,
                "Ground Truth": self.samplewise_real,
                "MAE": self.calc_error(),
            }
        )
        errors_df.to_csv(path)

    def reset_performance(self):
        self.samplewise_predicted = []
        self.samplewise_real = []

        self.loss = []
        self.mae = []
        self.crossval_rs = []
        self.crossval_scores = []

        self.histories = []
        self.predicted_test_values = []

        self.last_crossval_fit_time = None
        self.last_crossval_predict_time = None

    def calculate_fold(self, data: np.ndarray, labels, i, folds=5):
        val_data = data[i::folds]
        train_data = [
            data[index::folds] if not i == index else [] for index in range(folds)
        ]
        train_data = np.array(
            [element for sublist in train_data for element in sublist]
        )
        val_labels = labels[i::folds]
        train_labels = [
            labels[index::folds] if not i == index else [] for index in range(folds)
        ]
        train_labels = np.array(
            [element for sublist in train_labels for element in sublist]
        )
        return train_data, train_labels, val_data, val_labels

    def cross_val(
        self,
        data: np.ndarray,
        labels,
        prepare_function,
        test_data,
        folds=5,
    ):
        self.reset_performance()
        predict_times = []
        fit_times = []

        for i in range(folds):
            gc.collect()
            time.sleep(10)
            gc.collect()
            # Calculate data for current fold
            train_data, train_labels, val_data, val_labels = self.calculate_fold(
                data, labels, i, folds
            )

            self.model = self.get_model(train_labels)
            # Prepare, fit, evaluate
            train_data, val_data = prepare_function(
                train_data, False
            ), prepare_function(val_data, True)
            elapsed_time = self.fit(
                train_data, val_data, train_labels, val_labels, continue_training=True
            )
            fit_times.append(elapsed_time)
            loss, score, mae, r2 = self.evaluate(val_data, val_labels, train_labels)
            val_prediction = self.predict(val_data)
            start_time = time.time()
            test_prediction = self.predict(prepare_function(test_data, True))
            predict_times.append(time.time() - start_time)

            # calculate data for sample error report
            iterator_element = i
            index_element = 0
            while iterator_element < len(data):
                self.samplewise_predicted.insert(
                    iterator_element, val_prediction[index_element]
                )
                self.samplewise_real.insert(iterator_element, val_labels[index_element])

                iterator_element += folds
                index_element += 1

            # calculate data for performance report
            self.predicted_test_values.append(test_prediction)
            self.crossval_scores.append(score)
            self.crossval_rs.append(r2)
            self.loss.append(loss)
            self.mae.append(mae)

        self.predicted_test_values = np.array(self.predicted_test_values).mean(axis=0)
        self.crossval_scores = np.array(self.crossval_scores)
        self.crossval_rs = np.array(self.crossval_rs)
        self.loss = np.array(self.loss)
        self.mae = np.array(self.mae)
        self.last_crossval_predict_time = np.array(predict_times).mean()
        self.last_crossval_fit_time = np.array(fit_times).mean()
