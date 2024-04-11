
import os

import numpy as np
import tensorflow.keras as keras
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import cv2

import pandas as pd

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
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=270, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.RandomRotate90(p=0.75),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.15, 0.15), p=0.75),
            # A.RandomGamma(p=0.5),
            A.GaussNoise(var_limit=(0.0, 200.0), p=0.75),
            A.Defocus(radius=(1, 3), p=0.75)
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
            X[i,] = self.read_image(os.path.join(self.img_path, ID))
            y[i] = self.read_image(os.path.join(self.mask_path, ID.replace('.JPG', '.tif')))

            if self.augmentation:
                augmented = self.transform(image=X[i,], mask=y[i,])
                X[i,] = augmented['image']
                y[i,] = augmented['mask']

        X = np.expand_dims(X, axis=3)
        y = np.expand_dims(y, axis=3).astype(bool)

        return X, y
    

class ClassificationDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, img_path, label_df, batch_size=32, dim=(800,800), n_channels=1,
                 n_classes=1, shuffle=True, augmentation=True, mask_path=None, mode=1):
        'Initialization'
        self.dim = (*dim, n_channels)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_path = img_path
        self.labels = label_df
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=270, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.RandomRotate90(p=0.75),
            # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.15, 0.15), p=0.75),
            # A.RandomGamma(p=0.5),
            A.GaussNoise(var_limit=(0, 200), p=0.75),
            # A.Defocus(radius=(1, 3), p=0.75)
        ])
        self.mode = mode
        self.mask_path = mask_path
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
    
    def center_image_and_mask(self, image, mask):
        """Centers the image and mask while keeping the original image size."""

        # Use the crop_around_mask function to find the bounding box of the mask
        rows, cols = np.where(mask > 0)
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1

        # Calculate the center of the bounding box
        center_y = (ymin + ymax) // 2
        center_x = (xmin + xmax) // 2

        # Calculate the center of the original image
        image_center_y = image.shape[0] // 2
        image_center_x = image.shape[1] // 2

        # Calculate the offset needed to move the bounding box to the center of the image
        offset_y = image_center_y - center_y
        offset_x = image_center_x - center_x

        # Create new arrays for the centered image and mask
        centered_image = np.zeros_like(image)
        centered_mask = np.zeros_like(mask)

        # Apply the offset to the image and mask
        centered_image[max(0, offset_y):min(image.shape[0], image.shape[0] + offset_y),
                    max(0, offset_x):min(image.shape[1], image.shape[1] + offset_x)] = \
            image[max(0, -offset_y):min(image.shape[0], image.shape[0] - offset_y),
                max(0, -offset_x):min(image.shape[1], image.shape[1] - offset_x)]

        centered_mask[max(0, offset_y):min(mask.shape[0], mask.shape[0] + offset_y),
                    max(0, offset_x):min(mask.shape[1], mask.shape[1] + offset_x)] = \
            mask[max(0, -offset_y):min(mask.shape[0], mask.shape[0] - offset_y),
                max(0, -offset_x):min(mask.shape[1], mask.shape[1] - offset_x)]

        return centered_image, centered_mask

    # def pad_to_original_size(self, image, original_shape):
    #     """Pads the image with zeros to match the original shape."""
    #     missing_rows = original_shape[0] - image.shape[0]
    #     missing_cols = original_shape[1] - image.shape[1]
    #     padded_image = np.pad(image, ((0, missing_rows), (0, missing_cols), (0, 0)), mode='constant')
    #     return padded_image


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y = np.empty(self.batch_size, dtype=float)

        if self.mode != 1:
            X_mask = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        if self.mode == 1:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Read image and mask
                img = self.read_image(os.path.join(self.img_path, str(int(ID)) + '.jpg'))
                # X[i,] = np.stack((img,)*3, axis=-1)
                X[i,] = np.expand_dims(img, -1)
                y[i] = self.labels.loc[ID, 'outcome']

                if self.augmentation:
                    augmented = self.transform(image=X[i,])
                    X[i,] = augmented['image']

            
            return X, y
            
        elif self.mode == 2:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Read image and mask
                img = self.read_image(os.path.join(self.img_path, str(int(ID)) + '.jpg'))
                mask = self.read_image(os.path.join(self.mask_path, str(int(ID)) + '_mask.tif'))
                # X[i,] = np.stack((img,)*3, axis=-1)
                X[i,] = np.expand_dims(img, -1)
                X_mask[i] = np.expand_dims(mask, -1)

                y[i] = self.labels.loc[ID, 'outcome']

                if self.augmentation:
                    augmented = self.transform(image=X[i,], mask=X_mask[i])
                    X[i,] = augmented['image']
                    X_mask[i,] = augmented['mask']

                # ELSE!

            plt.imshow(X_mask[0,], cmap='gray')
            plt.show()

            return [X, X_mask], y
        
        elif self.mode == 3:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Read image and mask
                img = self.read_image(os.path.join(self.img_path, str(int(ID)) + '.jpg'))
                mask = (self.read_image(os.path.join(self.mask_path, str(int(ID)) + '_mask.tif')) / 255).astype(np.uint8)

                y[i] = self.labels.loc[ID, 'outcome']

                if self.augmentation:
                    augmented = self.transform(image=img, mask=mask)
                    # X[i,] = np.expand_dims(augmented['image'] * augmented['mask'], -1)

                        # Crop around the mask
                    img_cropped, _ = self.center_image_and_mask(augmented['image'] * augmented['mask'], augmented['mask'])
                    
                    # # pad 10 pixels around the cropped image
                    # img_cropped = cv2.copyMakeBorder(img_cropped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

                    # # Pad to original size if necessary
                    # img_cropped = cv2.resize(img_cropped, (self.dim[0], self.dim[1]))
                
                    # img_cropped = augmented['image'] * augmented['mask']

                    X[i,] = np.expand_dims(img_cropped, -1)
                else:
                    img, mask = self.center_image_and_mask(img, mask)
                    X[i,] = np.expand_dims(img*mask, -1)
            
                # ELSE!

            return X, y
    

if __name__ == "__main__":
    IMG_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\Prediction/"
    df_path = r"C:\Users\koenk\OneDrive\Technical Medicine\Jaar 3\04 Master Thesis\inclusions for blst expansion project_werkversie.xlsx"

    df_label = pd.read_excel(df_path).set_index('Embryo_id')
    df_label = df_label.drop(df_label[df_label['Included']==0].index)

    df_label['label'] = df_label['outcome']

    datagen = ClassificationDataGenerator(list_IDs=df_label.index, img_path=IMG_PATH, shuffle=False, augmentation=False, label_df=df_label, batch_size=8, dim=(800, 800), n_channels=1, mode=3, mask_path=IMG_PATH+'masks/')

    X, y = datagen.__getitem__(0)
    
    fig, axs = plt.subplots(2,4)

    for i in range(len(axs.ravel())):

        axs.ravel()[i].imshow(X[i,], cmap='gray')
        axs.ravel()[i].set_title(f"Label: {y[i]}")
        # axs.ravel()[i].imshow(y[i,], cmap='jet', interpolation='nearest', alpha=0.5)

    plt.tight_layout()
    # plt.imshow(X[0,], cmap='gray')
    # plt.imshow(y[0,], cmap='jet', interpolation='nearest', alpha=0.5)
    plt.show()
