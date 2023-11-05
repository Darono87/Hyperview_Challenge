from typing import List

import tensorflow as tf

from creators.ModelParams import (
    DeepLayerParams,
    NeuralNetworkFlowMixin,
    RecurrentLayerParams,
    GenericNeuralNetworkParams,
    FlattenParams,
)
from creators.SubCreators import create_deep, create_flatten, create_recurrent


class RNNNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams,
        rec: List[RecurrentLayerParams] = [RecurrentLayerParams()],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.rec = rec
        self.deep = deep
        self.generic = generic
        self.flatten = flatten

    def builder(self):
        model = tf.keras.models.Sequential()
        create_recurrent(self.rec, model)
        create_flatten(self.flatten, model)
        create_deep(self.deep, model)

        model.add(tf.keras.layers.Dense(1))
        return model
