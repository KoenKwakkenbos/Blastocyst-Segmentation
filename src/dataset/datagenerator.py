
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
                 n_classes=1, shuffle=True, augmentation=True, mask_path=None, mode=1, feature_df=None):
        'Initialization'
        self.dim = (*dim, n_channels)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_path = img_path
        self.labels = label_df
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        if feature_df is not None:
            self.feature_df = pd.read_csv(feature_df).set_index('Unnamed: 0').drop('label_p', axis=1).drop('label_i', axis=1)
            self.feature_df = self.feature_df[self.feature_df.columns[:17]]
            # TODO remove this
        else:
            self.feature_df = None
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            A.Rotate(limit=270, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            # A.RandomRotate90(p=0.75),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.75),
            # A.RandomGamma(p=0.5),
            # A.GaussNoise(var_limit=(0, 200), p=0.75),
            A.Defocus(radius=(1, 3), p=0.75)
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
    
    def center_image_and_mask(self, image, mask, scale=True):
        """Centers the image and mask while keeping the original image size."""

        # Use the crop_around_mask function to find the bounding box of the mask
        rows, cols = np.where(mask > 0)
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1

        # Calculate the center of the bounding box
        center_y = (ymin + ymax) // 2
        center_x = (xmin + xmax) // 2

        if scale:
            ymin = max(ymin - 10, 0)
            ymax = min(ymax + 10, self.dim[0])
            xmin = max(xmin - 10, 0)
            xmax = min(xmax + 10, self.dim[1])

            new_img = image[ymin:ymax, xmin:xmax]
            new_mask = mask[ymin:ymax, xmin:xmax]
    
            # rescale back to original size
            new_img = cv2.resize(new_img, (self.dim[0], self.dim[1]))
            new_mask = cv2.resize(new_mask, (self.dim[0], self.dim[1]))

            return new_img, new_mask
        else:
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
        X = np.empty((self.batch_size, *self.dim), dtype=float)
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

            if self.feature_df is not None:
                features = self.feature_df.loc[np.array(list_IDs_temp)].values
                return [X, features], y
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
            if self.feature_df is not None:
                features = self.feature_df.loc[np.array(list_IDs_temp)].values
                return [X, features], y
            return [X, X_mask], y
        
        elif self.mode == 3:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                # Read image and mask
                img = self.read_image(os.path.join(self.img_path, str(int(ID)) + '.jpg'))
                mask = (self.read_image(os.path.join(self.mask_path, str(int(ID)) + '_mask.tif')) / 255).astype(np.uint8)

                y[i] = self.labels.loc[ID, 'outcome']

                if self.augmentation:
                    # X[i,] = np.expand_dims(augmented['image'] * augmented['mask'], -1)

                        # Crop around the mask
                    img_cropped, mask_cropped = self.center_image_and_mask(img * mask, mask)
                    
                    augmented = self.transform(image=img_cropped, mask=mask_cropped)
                    # # pad 10 pixels around the cropped image 
                    # img_cropped = cv2.copyMakeBorder(img_cropped, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

                    # # Pad to original size if necessary
                    # img_cropped = cv2.resize(img_cropped, (self.dim[0], self.dim[1]))
                
                    # min_max norm
                    # img = cv2.normalize(augmented['image'], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F, mask=augmented['mask'])

                    X[i,] = np.expand_dims(img, -1)
                else:
                    img, mask = self.center_image_and_mask(img, mask)
                    # img = cv2.normalize(img*mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F, mask=mask)
                    X[i,] = np.expand_dims(img, -1)
            
                # ELSE!

            if self.feature_df is not None:
                features = self.feature_df.loc[np.array(list_IDs_temp)].values
                return [X, features], y
            return X, y
    

if __name__ == "__main__":
    IMG_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\Prediction/"
    df_path = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\Blast_labels.csv"

    df_label = pd.read_csv(df_path).set_index('ID')

    datagen = ClassificationDataGenerator(list_IDs=df_label.index, img_path=IMG_PATH, shuffle=False, augmentation=True, label_df=df_label, batch_size=8, dim=(800, 800), n_channels=1, mode=3, mask_path=IMG_PATH+'masks/')

    for i in range(20):
        X, y = datagen.__getitem__(i)
    
    fig, axs = plt.subplots(2,4)

    for i in range(len(axs.ravel())):

        axs.ravel()[i].imshow(X[i,], cmap='gray')
        axs.ravel()[i].set_title(f"Label: {y[i]}")
        # axs.ravel()[i].imshow(y[i,], cmap='jet', interpolation='nearest', alpha=0.5)
    plt.tight_layout()
    plt.show()

    df = pd.read_csv(r"C:\Users\koenk\Documents\Master_Thesis\Programming\Blastocyst-Segmentation\features.csv").set_index('Unnamed: 0')
    
    # datagen = ClassificationDataGenerator(list_IDs=df_label.index, img_path=IMG_PATH, shuffle=False, augmentation=False, label_df=df_label, batch_size=8, dim=(800, 800), n_channels=1, mode=3, mask_path=IMG_PATH+'masks/',
                                        #   feature_df = r"C:\Users\koenk\Documents\Master_Thesis\Programming\Blastocyst-Segmentation\features.csv")

    datagen = ClassificationDataGenerator(list_IDs=[1, 16, 18, 21, 32, 33, 35, 39], img_path=IMG_PATH, shuffle=False, augmentation=False, label_df=df_label, batch_size=8, dim=(800, 800), n_channels=1, mode=3, mask_path=IMG_PATH+'masks/',
                                          feature_df = r"C:\Users\koenk\Documents\Master_Thesis\Programming\Blastocyst-Segmentation\features.csv")

    X, y = datagen.__getitem__(0)
    
    model = tf.keras.models.load_model('C:/users/koenk/Downloads/model_fold_1.h5', compile=False)
    preds = model.predict(X)

    fig, axs = plt.subplots(2,4)

    for i in range(len(axs.ravel())):

        axs.ravel()[i].imshow(X[0][i,], cmap='gray')
        axs.ravel()[i].set_title(f"Prediction: {preds[i][0]:.2f}, Label: {y[i]}")
        # axs.ravel()[i].imshow(y[i,], cmap='jet', interpolation='nearest', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(features)
