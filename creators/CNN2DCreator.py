from typing import List
import tensorflow as tf
from creators.ModelParams import (
    FlattenParams,
    Conv2DParams,
    GenericNeuralNetworkParams,
    DeepLayerParams,
    NeuralNetworkFlowMixin,
)
from creators.CNNCreator import CNNCreator


class CNN2DNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams,
        conv: List[Conv2DParams] = [Conv2DParams()],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.conv = conv
        self.generic = generic
        self.deep = deep
        self.flatten = flatten

    def builder(self):
        return CNNCreator(
            self,
            tf.keras.layers.Conv2D,
            tf.keras.layers.SpatialDropout2D,
            tf.keras.layers.MaxPool2D,
        )
