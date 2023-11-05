from typing import Callable, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
import skops.io as sio
from flow.ClassicalModelFlow import ClassicalModelFlow
from flow.ModelFlow import calc_partial_score
from flow.NeuralNetworkFlow import NeuralNetworkFlow


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class ClassicalModelFlowMixin:
    def generate_flow(self, model_name):
        if self.load_from != None:
            with open(self.load_from, "rb") as binary_file:
                content = binary_file.read()
                return ClassicalModelFlow(lambda _: sio.loads(content, trusted=True))

        return ClassicalModelFlow(lambda _: self.builder())


class NeuralNetworkFlowMixin:
    def generate_flow(self, model_name="model"):
        def compile_neural_network(train_gt):
            if hasattr(compile_neural_network, "num_gen_model"):
                compile_neural_network.num_gen_model += 1
            else:
                compile_neural_network.num_gen_model = 0

            def score_metric(gt, pred):
                return calc_partial_score(train_gt, pred.numpy(), gt.numpy())

            metrics = [
                tf.keras.metrics.MeanAbsoluteError(),
                score_metric,
                get_lr_metric(self.generic.optimizer()),
            ]

            if self.generic.load_from != None:
                new_model = tf.keras.models.load_model(
                    self.generic.load_from,
                    compile=False,
                    custom_objects={"lr": metrics[2], "score_metric": metrics[1]},
                )
            else:
                new_model = self.builder()

            new_model.compile(
                run_eagerly=True,
                optimizer=self.generic.optimizer(),
                loss=self.generic.loss,
                metrics=metrics,
            )

            return new_model

        flow = NeuralNetworkFlow(
            compile_neural_network,
            epochs=self.generic.epochs,
            get_callbacks=lambda: [
                tf.keras.callbacks.History(),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.6, patience=10, min_lr=0.00001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="experiment1/checkpoints/"
                    + str(model_name)
                    + "-"
                    + str(compile_neural_network.num_gen_model)
                    + "-{epoch:02d}-{val_score_metric:.2f}.h5",
                    monitor="val_score_metric",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="min",
                    save_freq="epoch",
                    initial_value_threshold=1.5,
                ),
            ],
        )
        return flow


class GenericNeuralNetworkParams:
    def __init__(
        self,
        prepare_function: Callable[[List[np.ndarray], bool], np.ndarray],
        optimizer,
        epochs=50,
        loss=tf.losses.mae,
        load_from=None,
    ):
        self.prepare_function = prepare_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = loss
        self.load_from = load_from


class FlattenParams:
    def __init__(
        self,
        activation=None,
        dropout=0,
        normalization=False,
    ):
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization


class DeepLayerParams:
    def __init__(
        self,
        neurons=32,
        activation=None,
        dropout=0,
        normalization=False,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
    ):
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg


class Conv1DParams:
    def __init__(
        self,
        filters=64,
        kernel=5,
        pool_size=3,
        dropout=0,
        normalization=False,
        activation=None,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        noise=0,
    ):
        self.filters = filters
        self.kernel = kernel
        self.pool_size = pool_size
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg
        self.noise = noise


class Conv2DParams:
    def __init__(
        self,
        filters=150,
        kernel=(2, 2),
        pool_size=(2, 2),
        dropout=0,
        normalization=False,
        activation=None,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        noise=0,
    ):
        self.filters = filters
        self.kernel = kernel
        self.pool_size = pool_size
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg
        self.noise = noise


class Conv3DParams:
    def __init__(
        self,
        filters=32,
        kernel=(2, 2, 2),
        pool_size=(2, 2, 2),
        dropout=0,
        normalization=False,
        activation=None,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        noise=0,
    ):
        self.filters = filters
        self.kernel = kernel
        self.pool_size = pool_size
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg
        self.noise = noise


class RecurrentLayerParams:
    def __init__(
        self,
        type=tf.keras.layers.LSTM,
        units=32,
        dropout=0,
        recurrent_dropout=0.0,
        normalization=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        bidirectional=False,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        noise=0,
    ):
        self.recurrent_activation = recurrent_activation
        self.recurrent_dropout = recurrent_dropout
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.bidirectional = bidirectional
        self.type = type
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg
        self.noise = noise


class ConvolutionalRecurrentLayerParams:
    def __init__(
        self,
        filters=32,
        kernel_size=(3, 3),
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        strides=(1, 1),
        dropout=0.0,
        recurrent_dropout=0.0,
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        noise=0,
        normalization=False,
        bidirectional=False,
    ):
        self.strides = strides
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_activation = recurrent_activation
        self.filters = filters
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.kernel_size = kernel_size
        self.type = type
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
        self.act_reg = act_reg
        self.noise = noise
        self.bidirectional = bidirectional


class RandomForestParams(ClassicalModelFlowMixin):
    def __init__(
        self,
        prepare_function: Callable[[List[np.ndarray], bool], np.ndarray],
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=1.0,
        load_from=None,
    ):
        self.prepare_function = prepare_function
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.load_from = load_from

    def builder(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
        )


class SVRParams(ClassicalModelFlowMixin):
    def __init__(
        self,
        prepare_function: Callable[[List[np.ndarray], bool], np.ndarray],
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        C=1.0,
        load_from=None,
    ):
        self.prepare_function = prepare_function
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.load_from = load_from

    def builder(self):
        return SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            C=self.C,
        )
