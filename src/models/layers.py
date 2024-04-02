# src/models/layers.py
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, BatchNormalization, Activation, Multiply, Permute, Conv2D
import tensorflow as tf


def squeeze_excite_block(input: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False)(se)
    se = BatchNormalization()(se)
    se = Activation(tf.keras.activations.gelu)(se)
    se = Dense(filters, kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation('sigmoid')(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x


def cbam_block(input_tensor: tf.Tensor, ratio: int = 16, kernel_size: int = 7) -> tf.Tensor:
    channel = GlobalAveragePooling2D()(input_tensor)
    channel = Dense(channel.shape[-1] // ratio, kernel_initializer='he_normal', use_bias=True)(channel)
    channel = BatchNormalization()(channel)
    channel = Activation(tf.keras.activations.gelu)(channel)
    channel = Dense(input_tensor.shape[-1], kernel_initializer='he_normal', use_bias=True)(channel)
    channel = Activation('sigmoid')(channel)

    if K.image_data_format() == 'channels_first':
        channel = Reshape((input_tensor.shape[1], 1, 1))(channel)
    else:  # 'channels_last'
        channel = Reshape((1, 1, input_tensor.shape[-1]))(channel)

    channel_attention = Multiply()([input_tensor, channel])

    spatial = Conv2D(1, kernel_size, padding='same', use_bias=True)(channel_attention)
    spatial = Activation('sigmoid')(spatial)
    output = Multiply()([channel_attention, spatial])

    return output
