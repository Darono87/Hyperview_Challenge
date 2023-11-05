from typing import List

import tensorflow as tf
from creators.ModelParams import (
    FlattenParams,
    GenericNeuralNetworkParams,
    ConvolutionalRecurrentLayerParams,
    DeepLayerParams,
    NeuralNetworkFlowMixin,
)
from creators.SubCreators import create_deep, create_flatten


class CLSTMNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams = FlattenParams(),
        rec: List[ConvolutionalRecurrentLayerParams] = [
            ConvolutionalRecurrentLayerParams()
        ],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.rec = rec
        self.deep = deep
        self.generic = generic
        self.flatten = flatten

    def builder(self):
        model = tf.keras.models.Sequential()

        for index, rec in enumerate(self.rec):
            if rec.dropout > 0:
                model.add(tf.keras.layers.SpatialDropout3D(rec.dropout))
            rec_layer = tf.keras.layers.ConvLSTM2D(
                filters=rec.filters,
                kernel_size=rec.kernel_size,
                activation=rec.activation,
                recurrent_activation=rec.recurrent_activation,
                strides=rec.strides,
                bias_regularizer=rec.bias_reg,
                kernel_regularizer=rec.kernel_reg,
                activity_regularizer=rec.act_reg,
                return_sequences=index < len(self.rec) - 1,
            )
            model.add(rec_layer)
            if rec.normalization:
                model.add(tf.keras.layers.BatchNormalization())
            if rec.noise > 0:
                model.add(tf.keras.layers.GaussianNoise(rec.noise))

        create_flatten(self.flatten, model)
        create_deep(self.deep, model)

        model.add(tf.keras.layers.Dense(1))
        return model
