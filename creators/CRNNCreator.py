from typing import List

import tensorflow as tf

from creators.ModelParams import (
    FlattenParams,
    GenericNeuralNetworkParams,
    NeuralNetworkFlowMixin,
    RecurrentLayerParams,
    Conv3DParams,
    DeepLayerParams,
)
from creators.SubCreators import (
    create_convolutional,
    create_deep,
    create_flatten,
    create_recurrent,
)


class CRNNNetworkParams(NeuralNetworkFlowMixin):
    def __init__(
        self,
        generic: GenericNeuralNetworkParams,
        flatten: FlattenParams = FlattenParams(),
        rec: List[RecurrentLayerParams] = [RecurrentLayerParams()],
        conv: List[Conv3DParams] = [Conv3DParams()],
        deep: List[DeepLayerParams] = [DeepLayerParams()],
    ):
        self.rec = rec
        self.deep = deep
        self.conv = conv
        self.generic = generic
        self.flatten = flatten

    def builder(self):
        model = tf.keras.models.Sequential()

        submodel = tf.keras.models.Sequential()

        create_convolutional(
            self.conv,
            tf.keras.layers.Conv3D,
            tf.keras.layers.SpatialDropout3D,
            tf.keras.layers.MaxPool3D,
            submodel,
        )

        create_flatten(self.flatten, submodel)

        model.add(tf.keras.layers.TimeDistributed(submodel))

        create_recurrent(self.rec, model)

        create_deep(self.deep, model)

        model.add(tf.keras.layers.Dense(1))
        return model
