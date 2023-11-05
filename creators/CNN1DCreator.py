from typing import List
import tensorflow as tf
from creators.ModelParams import (
    Conv1DParams,
    FlattenParams,
    GenericNeuralNetworkParams,
    DeepLayerParams,
    NeuralNetworkFlowMixin,
)
from creators.CNNCreator import CNNCreator


class CNN1DNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams,
        conv: List[Conv1DParams] = [Conv1DParams()],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.generic = generic
        self.conv = conv
        self.deep = deep
        self.flatten = flatten

    def builder(self):
        return CNNCreator(
            self,
            tf.keras.layers.Conv1D,
            tf.keras.layers.SpatialDropout1D,
            tf.keras.layers.MaxPool1D,
        )
