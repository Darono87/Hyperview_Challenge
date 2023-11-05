from typing import List
import tensorflow as tf
from creators.ModelParams import (
    Conv3DParams,
    FlattenParams,
    GenericNeuralNetworkParams,
    DeepLayerParams,
    NeuralNetworkFlowMixin,
)
from creators.CNNCreator import CNNCreator


class CNN3DNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams = FlattenParams(),
        conv: List[Conv3DParams] = [Conv3DParams()],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.generic = generic
        self.conv = conv
        self.deep = deep
        self.flatten = flatten

    def builder(self):
        return CNNCreator(
            self,
            tf.keras.layers.Conv3D,
            tf.keras.layers.SpatialDropout3D,
            tf.keras.layers.MaxPool3D,
        )
