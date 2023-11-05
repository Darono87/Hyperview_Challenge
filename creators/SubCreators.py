from typing import List
import tensorflow as tf
from creators.ModelParams import DeepLayerParams, FlattenParams, RecurrentLayerParams


def create_flatten(flatten: FlattenParams, model):
    model.add(tf.keras.layers.Flatten())
    if flatten.activation:
        model.add(tf.keras.layers.Activation(flatten.activation))
    if flatten.normalization:
        model.add(tf.keras.layers.BatchNormalization())
    if flatten.dropout:
        model.add(tf.keras.layers.Dropout(flatten.dropout))


def create_deep(deep: List[DeepLayerParams], model):
    for deep in deep:
        model.add(
            tf.keras.layers.Dense(
                deep.neurons,
                bias_regularizer=deep.bias_reg,
                kernel_regularizer=deep.kernel_reg,
                activity_regularizer=deep.act_reg,
                activation=deep.activation,
            )
        )
        if deep.normalization:
            model.add(tf.keras.layers.BatchNormalization())
        if deep.dropout != 0:
            model.add(tf.keras.layers.Dropout(deep.dropout))


def create_recurrent(recs: List[RecurrentLayerParams], model):
    for index, rec in enumerate(recs):
        rec_layer = rec.type(
            activation=rec.activation,
            recurrent_activation=rec.recurrent_activation,
            bias_regularizer=rec.bias_reg,
            kernel_regularizer=rec.kernel_reg,
            activity_regularizer=rec.act_reg,
            units=rec.units,
            dropout=rec.dropout,
            recurrent_dropout=rec.recurrent_dropout,
            return_sequences=index < len(recs) - 1,
        )
        model.add(
            tf.keras.layers.Bidirectional(rec_layer) if rec.bidirectional else rec_layer
        )
        if rec.normalization:
            model.add(tf.keras.layers.BatchNormalization())
        if rec.noise > 0:
            model.add(tf.keras.layers.GaussianNoise(rec.noise))


def create_convolutional(conv, convLayer, dropoutLayer, poolingLayer, model):
    for conv in conv:
        model.add(
            convLayer(
                filters=conv.filters,
                kernel_size=conv.kernel,
                padding="same",
                activation=conv.activation,
                bias_regularizer=conv.bias_reg,
                kernel_regularizer=conv.kernel_reg,
                activity_regularizer=conv.act_reg,
            )
        )
        if conv.normalization:
            model.add(tf.keras.layers.BatchNormalization())
        if conv.dropout != 0:
            model.add(dropoutLayer(conv.dropout))
        if conv.noise > 0:
            model.add(tf.keras.layers.GaussianNoise(conv.noise))
        model.add(poolingLayer(pool_size=conv.pool_size))
