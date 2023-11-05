import tensorflow as tf

from creators.SubCreators import create_convolutional, create_deep, create_flatten


def CNNCreator(metaparams, convLayer, dropoutLayer, poolingLayer):
    model = tf.keras.models.Sequential()
    create_convolutional(metaparams.conv, convLayer, dropoutLayer, poolingLayer, model)
    create_flatten(metaparams.flatten, model)
    create_deep(metaparams.deep, model)
    model.add(tf.keras.layers.Dense(1))
    return model
