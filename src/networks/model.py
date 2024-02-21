"""
Author: Koen Kwakkenbos
Date: 2024-02-01

This module contains a function to build a U-Net model using TensorFlow and Keras.

Functions:
    build_unet_model(input_height, input_width, input_channels): Builds a U-Net model.
"""

import tensorflow as tf

def build_unet_model(input_height, input_width, input_channels, print_summary=True):
    """Builds a U-Net model"""

    # Define input layer
    inputs = tf.keras.layers.Input((input_height, input_width, input_channels))

    # Normalize input data
    normalized_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Down sampling path
    conv1 = tf.keras.layers.Conv2D(16, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(normalized_inputs)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    conv5 = tf.keras.layers.Conv2D(256, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    pool5 = tf.keras.layers.MaxPooling2D((2, 2))(conv5)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool5)

    # Upsampling path
    up7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = tf.keras.layers.concatenate([up7, conv5])
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat7)
    up8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = tf.keras.layers.concatenate([up8, conv4])
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat8)
    up9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = tf.keras.layers.concatenate([up9, conv3])
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat9)
    up10 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9)
    concat10 = tf.keras.layers.concatenate([up10, conv2])
    conv10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat10)
    up11 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv10)
    concat11 = tf.keras.layers.concatenate([up11, conv1], axis=3)
    conv11 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat11)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    # Build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    if print_summary:
        print(model.summary())

    return model


def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    conv = tf.keras.layers.Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = tf.keras.layers.BatchNormalization(axis=3)(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    conv = tf.keras.layers.Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = tf.keras.layers.BatchNormalization(axis=3)(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    return conv

def UNet(input_shape=(256, 256, 1), num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    conv1 = double_conv_layer(inputs, 32)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = double_conv_layer(pool1, 64)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = double_conv_layer(pool2, 128)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = double_conv_layer(pool3, 256)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = double_conv_layer(pool4, 512)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = double_conv_layer(up6, 256)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = double_conv_layer(up7, 128)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = double_conv_layer(up8, 64)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = double_conv_layer(up9, 32)

    conv10 = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    print(model.summary())

    return model


def build_rd_unet(input_height, input_width, input_channels, print_summary=True):
    """Builds a Residual Dilated U-Net model"""

    # Define input layer
    inputs = tf.keras.layers.Input((input_height, input_width, input_channels))

    # Normalize input data
    normalized_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Down sampling path
    conv1 = tf.keras.layers.Conv2D(8, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(normalized_inputs)
    conv1 = tf.keras.layers.Conv2D(8, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv1)
    skip1 = tf.keras.layers.Conv2D(8, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(normalized_inputs)
    conv1 = tf.keras.layers.Add()([conv1, skip1])
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(16, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(16, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv2)
    skip2 = tf.keras.layers.Conv2D(16, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Add()([conv2, skip2])
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(32, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(32, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv3)
    skip3 = tf.keras.layers.Conv2D(32, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Add()([conv3, skip3])
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(48, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(48, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv4)
    skip4 = tf.keras.layers.Conv2D(48, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Add()([conv4, skip4])
    conv4 = tf.keras.layers.Activation('relu')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    # bridge
    bridge1 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=1, activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    bridge2 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same')(bridge1)
    bridge3 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same')(bridge2)
    bridge4 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same')(bridge3)
    bridge5 = tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=16, activation='relu', kernel_initializer='he_normal', padding='same')(bridge4)
    bridge6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bridge5)

    # Upsampling path
    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(bridge6)
    up8 = tf.keras.layers.concatenate([up8, conv4])
    conv8 = tf.keras.layers.Conv2D(48, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(48, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv8)
    skip8 = tf.keras.layers.Conv2D(48, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up8)
    conv8 = tf.keras.layers.Add()([conv8, skip8])
    conv8 = tf.keras.layers.Activation('relu')(conv8)

    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)

    up9 = tf.keras.layers.concatenate([up9, conv3])
    conv9 = tf.keras.layers.Conv2D(32, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv9)
    skip9 = tf.keras.layers.Conv2D(32, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up9)
    conv9 = tf.keras.layers.Add()([conv9, skip9])
    conv9 = tf.keras.layers.Activation('relu')(conv9)

    up10 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv9)

    up10 = tf.keras.layers.concatenate([up10, conv2])
    conv10 = tf.keras.layers.Conv2D(16, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up10)
    conv10 = tf.keras.layers.Conv2D(16, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv10)
    skip10 = tf.keras.layers.Conv2D(16, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up10)
    conv10 = tf.keras.layers.Add()([conv10, skip10])
    conv10 = tf.keras.layers.Activation('relu')(conv10)

    up11 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv10)

    up11 = tf.keras.layers.concatenate([up11, conv1])
    conv11 = tf.keras.layers.Conv2D(8, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up11)
    conv11 = tf.keras.layers.Conv2D(8, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv11)
    skip11 = tf.keras.layers.Conv2D(8, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up11)
    conv11 = tf.keras.layers.Add()([conv11, skip11])
    conv11 = tf.keras.layers.Activation('relu')(conv11)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    # Build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    if print_summary:
        print(model.summary())

    return model


if __name__ == '__main__':
    # model = build_unet_model(800, 800, 1)
    model = build_rd_unet(800, 800, 1)
