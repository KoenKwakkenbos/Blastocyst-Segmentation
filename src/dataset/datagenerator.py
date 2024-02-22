
import os

import numpy as np
import tensorflow.keras as keras
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import cv2

def read_image(path):
    """Reads image from path and returns image array"""
    img = skimage.io.imread(path, as_gray=True)

    return np.expand_dims(img, axis=2)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, img_path, mask_path, batch_size=32, dim=(800,800), n_channels=1,
                 n_classes=1, shuffle=True, augmentation=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_path = img_path
        self.mask_path = mask_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.75, border_mode=cv2.BORDER_REPLICATE),
            # A.Rotate(limit=270, p=0.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomRotate90(p=0.75),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.15, 0.15), p=0.75),
            # A.RandomGamma(p=0.5),
            # A.GaussNoise(var_limit=(50.0, 100.0), p=0.7),
        ])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_image(self, path):
        """Reads image from path and returns image array"""
        img = skimage.io.imread(path, as_gray=True)

        return img


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Read image and mask
            X[i,] = self.read_image(self.img_path + ID)
            y[i] = self.read_image(self.mask_path + ID.replace('.JPG', '.tif'))

            if self.augmentation:
                augmented = self.transform(image=X[i,], mask=y[i,])
                X[i,] = augmented['image']
                y[i,] = augmented['mask']

        X = np.expand_dims(X, axis=3)
        y = np.expand_dims(y, axis=3).astype(bool)

        return X, y
    

if __name__ == "__main__":
    IMG_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\images/"
    MASK_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\masks/"

    datagen = DataGenerator(list_IDs=os.listdir(IMG_PATH), img_path=IMG_PATH, mask_path=MASK_PATH, batch_size=8, dim=(800,800), n_channels=1)

    X, y = datagen.__getitem__(0)
    
    fig, axs = plt.subplots(2,4)

    for i in range(len(axs.ravel())):

        axs.ravel()[i].imshow(X[i,], cmap='gray')
        # axs.ravel()[i].imshow(y[i,], cmap='jet', interpolation='nearest', alpha=0.5)

    plt.tight_layout()
    # plt.imshow(X[0,], cmap='gray')
    # plt.imshow(y[0,], cmap='jet', interpolation='nearest', alpha=0.5)
    plt.show()
