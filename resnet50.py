'''
# Network Architecture (Resnet-50)
# Author: Zhihui Lu
# Date: 2019/05/12
'''

import tensorflow as tf
from octconv_2d import OctConv2D


def octconv_resnet50(x, alpha, is_training=True):
    residual_list = [3, 4, 6, 3]

    high = x
    low = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(x)

    # conv1 high
    high = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(high)
    high = batch_norm(high, is_training=is_training, scope='batch_norm_first_layer_high')
    high = tf.nn.relu(high)
    high = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(high)

    # conv1 low
    low = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(low)
    low = batch_norm(low, is_training=is_training, scope='batch_norm_first_layer_low')
    low = tf.nn.relu(low)
    low = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(low)

    # conv2_x
    for i in range(residual_list[0]):
        high, low = octconv_resblock([high, low], channels=64, alpha=alpha, index=i,
                                     is_training=is_training, scope='octconv_resblock2_' + str(i))

    # conv3_x
    high = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(high)
    low = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(low)

    for i in range(residual_list[1]):
        high, low = octconv_resblock([high, low], channels = 128, alpha=alpha, index=i,
                                     is_training = is_training, scope='octconv_resblock3_' + str(i))

    # conv4_x
    high = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(high)
    low = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(low)

    for i in range(residual_list[2]):
        high, low = octconv_resblock([high, low], channels = 256, alpha=alpha, index=i,
                                     is_training = is_training, scope='octconv_resblock4_' + str(i))

    # conv5_x
    for i in range(residual_list[3]):
        high, low = octconv_resblock([high, low], channels = 512, alpha=alpha, index=i,
                                     is_training = is_training, scope='octconv_resblock5_' + str(i))

    # concatenate high and low
    high = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(high)
    x = tf.keras.layers.concatenate([high, low])
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1,1), strides=(1,1))(x)
    x = batch_norm(x, is_training, scope='batch_norm_last_layer')
    x = tf.nn.relu(x)

    # FC
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(units=100, activation='softmax')(x)

    return output


def octconv_resblock(x_init, channels, alpha, index, is_training=True, scope='octconv_resblock'):
    with tf.variable_scope(scope) :
        high, low = x_init[0], x_init[1]

        high = batch_norm(high, is_training, scope='batch_norm_1x1_front_high')
        shortcut_high = tf.nn.relu(high)

        low = batch_norm(low, is_training, scope='batch_norm_1x1_front_low')
        shortcut_low = tf.nn.relu(low)


        high, low = OctConv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1),
                              alpha=alpha)([shortcut_high, shortcut_low])
        high = batch_norm(high, is_training, scope='batch_norm_3x3_high')
        high = tf.nn.relu(high)
        low = batch_norm(low, is_training, scope='batch_norm_3x3_low')
        low = tf.nn.relu(low)

        high, low = OctConv2D(filters=channels, kernel_size=(3, 3), strides=(1, 1), alpha=alpha)([high, low])

        high = batch_norm(high, is_training, scope='batch_norm_1x1_back_high')
        high = tf.nn.relu(high)
        low = batch_norm(low, is_training, scope='batch_nrom_1x1_back_low')
        low = tf.nn.relu(low)
        high, low = OctConv2D(filters=channels * 4, kernel_size=(1, 1), strides=(1, 1), alpha=alpha)([high, low])

        if index == 0:
            shortcut_high = tf.keras.layers.Conv2D(int(channels * 4 * (1 - alpha)), kernel_size=1, strides=1, padding='same')(shortcut_high)
            shortcut_low = tf.keras.layers.Conv2D(int(channels * 4 * alpha), kernel_size=1, strides=1, padding='same')(shortcut_low)

        high = tf.keras.layers.add([high, shortcut_high])
        low = tf.keras.layers.add([low, shortcut_low])

    return [high, low]

def normal_resnet50(x, alpha, is_training=True):
    residual_list = [3, 4, 6, 3]

    # conv1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(x)
    x = batch_norm(x, is_training=is_training, scope='batch_norm_first_layer')
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # conv2_x
    for i in range(residual_list[0]):
        x = normal_resblock(x, channels=64, is_training=is_training, scope='resblock2_' + str(i))

    # conv3_x
    x = normal_resblock(x, channels=128, is_training=is_training, downsample=True, scope='resblock3_0')

    for i in range(1, residual_list[1]):
        x = normal_resblock(x, channels = 128, is_training = is_training, scope='resblock3_' + str(i))

    # conv4_x
    x = normal_resblock(x, channels=256, is_training=is_training, downsample=True, scope='resblock4_0')

    for i in range(1, residual_list[2]):
        x = normal_resblock(x, channels=256, is_training=is_training, scope='resblock4_' + str(i))

    # conv5_x
    x = normal_resblock(x, channels=512, is_training=is_training, downsample=True, scope='resblock5_0')

    for i in range(1, residual_list[3]):
        x = normal_resblock(x, channels=512, is_training=is_training, scope='resblock5_' + str(i))

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(units=100, activation='softmax')(x)

    return output

def normal_resblock(x_init, channels, is_training=True, downsample=False, scope='bottle_resblock'):
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = tf.nn.relu(x)

        x = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1, padding='same')(shortcut)
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = tf.nn.relu(x)

        if downsample :
            x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, padding='same')(x)
            shortcut = tf.keras.layers.Conv2D(channels*4, kernel_size=1, strides=2, padding='same')(shortcut)

        else :
            x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding='same')(x)
            shortcut = tf.keras.layers.Conv2D(channels * 4, kernel_size=1, strides=1, padding='same')(shortcut)

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = tf.nn.relu(x)
        x = tf.keras.layers.Conv2D(channels*4, kernel_size=1, strides=1, padding='same')(x)

        return x + shortcut


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
