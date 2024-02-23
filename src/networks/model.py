"""
Author: Koen Kwakkenbos
Date: 2024-02-01

This module contains a function to build a U-Net model using TensorFlow and Keras.

Functions:
    build_unet(input_height, input_width, input_channels): Builds a U-Net model.
    build_rd_unet(input_height, input_width, input_channels): Builds a Residual Dilated U-Net model.
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Lambda, Add, Activation, UpSampling2D, Normalization, BatchNormalization
from tensorflow.keras import Model


def build_unet(input_shape=(800, 800, 1), filters=(16, 32, 64, 128, 256, 512), normalization='min_max', print_summary=True):
    """Builds a U-Net model.
    
    Parameters:
    ----------
    input_shape: tuple, default=(800, 800, 1)
        The shape of the input data.
    filters: tuple, default=(16, 32, 64, 128, 256, 512)
        The number of filters for each of the six layers.
    print_summary: bool, default=True
        Whether to print the model summary.
    """


    # Define input layer
    inputs = Input(shape=input_shape)

    # Normalize input data
    if normalization == 'min_max':
        normalized_inputs = Lambda(lambda x: x / 255)(inputs)
    elif normalization == 'batchnorm':
        normalized_inputs = BatchNormalization()(inputs)

    # Down sampling path
    conv1 = Conv2D(filters[0], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(normalized_inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(filters[1], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(filters[2], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = Conv2D(filters[3], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(filters[4], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    pool5 = MaxPooling2D((2, 2))(conv5)
    conv6 = Conv2D(filters[5], (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool5)

    # Upsampling path
    up7 = Conv2DTranspose(filters[5] // 2, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = concatenate([up7, conv5])
    conv7 = Conv2D(filters[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat7)
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = concatenate([up8, conv4])
    conv8 = Conv2D(filters[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat8)
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = concatenate([up9, conv3])
    conv9 = Conv2D(filters[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat9)
    up10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9)
    concat10 = concatenate([up10, conv2])
    conv10 = Conv2D(filters[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat10)
    up11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv10)
    concat11 = concatenate([up11, conv1], axis=3)
    conv11 = Conv2D(filters[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concat11)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    # Build the model
    model = Model(inputs=[inputs], outputs=[outputs])

    if print_summary:
        print(model.summary())

    return model


def build_rd_unet(input_shape=(800, 800, 1), normalization='min_max', print_summary=True):
    """Builds a Residual Dilated U-Net model.
    
    Parameters:
    ----------
    input_shape: tuple, default=(800, 800, 1)
        The shape of the input data.
    print_summary: bool, default=True
        Whether to print the model summary.
    """

    # Define input layer
    inputs = Input(input_shape)

    # Normalize input data
    if normalization == 'min_max':
        normalized_inputs = Lambda(lambda x: x / 255)(inputs)
    elif normalization == 'batchnorm':
        normalized_inputs = BatchNormalization()(inputs)

    # Down sampling path
    conv1 = Conv2D(8, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(normalized_inputs)
    conv1 = Conv2D(8, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv1)
    skip1 = Conv2D(8, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(normalized_inputs)
    conv1 = Add()([conv1, skip1])
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv2)
    skip2 = Conv2D(16, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Add()([conv2, skip2])
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv3)
    skip3 = Conv2D(32, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Add()([conv3, skip3])
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(48, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Conv2D(48, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv4)
    skip4 = Conv2D(48, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Add()([conv4, skip4])
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # bridge
    bridge1 = Conv2D(64, (3, 3), dilation_rate=1, activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    bridge2 = Conv2D(64, (3, 3), dilation_rate=2, activation='relu', kernel_initializer='he_normal', padding='same')(bridge1)
    bridge3 = Conv2D(64, (3, 3), dilation_rate=4, activation='relu', kernel_initializer='he_normal', padding='same')(bridge2)
    bridge4 = Conv2D(64, (3, 3), dilation_rate=8, activation='relu', kernel_initializer='he_normal', padding='same')(bridge3)
    bridge5 = Conv2D(64, (3, 3), dilation_rate=16, activation='relu', kernel_initializer='he_normal', padding='same')(bridge4)
    bridge6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bridge5)

    # Upsampling path
    up8 = UpSampling2D(size=(2, 2))(bridge6)
    up8 = concatenate([up8, conv4])
    conv8 = Conv2D(48, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Conv2D(48, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv8)
    skip8 = Conv2D(48, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Add()([conv8, skip8])
    conv8 = Activation('relu')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)

    up9 = concatenate([up9, conv3])
    conv9 = Conv2D(32, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv9)
    skip9 = Conv2D(32, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Add()([conv9, skip9])
    conv9 = Activation('relu')(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)

    up10 = concatenate([up10, conv2])
    conv10 = Conv2D(16, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up10)
    conv10 = Conv2D(16, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv10)
    skip10 = Conv2D(16, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up10)
    conv10 = Add()([conv10, skip10])
    conv10 = Activation('relu')(conv10)

    up11 = UpSampling2D(size=(2, 2))(conv10)

    up11 = concatenate([up11, conv1])
    conv11 = Conv2D(8, (3, 3),  activation='relu', kernel_initializer='he_normal', padding='same')(up11)
    conv11 = Conv2D(8, (3, 3),  activation=None, kernel_initializer='he_normal', padding='same')(conv11)
    skip11 = Conv2D(8, (1, 1),  activation=None, kernel_initializer='he_normal', padding='same')(up11)
    conv11 = Add()([conv11, skip11])
    conv11 = Activation('relu')(conv11)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    # Build the model
    model = Model(inputs=[inputs], outputs=[outputs])

    if print_summary:
        print(model.summary())

    return model


if __name__ == '__main__':
    print("This module contains functions to build U-Net models.")