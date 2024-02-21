import os

import numpy as np
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
import pandas as pd

from datetime import datetime
from PIL import Image

from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, recall_score
from skimage.morphology import disk, erosion, dilation
from skimage.measure import regionprops, label
from sklearn.model_selection import KFold

# Experiment variables
EXP_NUM = 7
EXP_NAME = 'experiment_' + f"{EXP_NUM:03}" + " (1)" # change back
TEST = True
IMG_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\images/"
IMG_TIMELAPSE_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\timelapse_data/"
MASK_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\masks/"
OUTPUT_PATH =  r"C:\Users\koenk\Documents\Master_Thesis\Programming\Debugging/"

# Create output directory for experiment
# check if path exists and create if not:
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, EXP_NAME)
if not os.path.exists(OUTPUT_FOLDER):
    print('Experiment folder:', OUTPUT_FOLDER, 'does not exist.')
    exit()
else:
    if os.listdir(OUTPUT_FOLDER) == []:
        print(OUTPUT_FOLDER, 'exists, but is empty.')
        exit()
    else:
        print('Generating growth curves for models in', OUTPUT_FOLDER)
        OUTPUT_CURVE_FOLDER = OUTPUT_FOLDER + '/' + 'Growth_curves' + '/'
        OUTPUT_IMAGE_FOLDER = OUTPUT_FOLDER + '/' + 'Segmented_images' + '/'
        os.makedirs(OUTPUT_CURVE_FOLDER, exist_ok=True)

# Image information
IMG_WIDTH = 800
IMG_HEIGHT = 800
IMG_CHANNELS = 1
IMG_PIXEL_SIZE = 0.34**2

# Model training
RANDOM_SEED = 1
BATCH_SIZE = 4
EPOCHS = 80
N_FOLDS = 4
LOSS_FN = 'binary_crossentropy'
OPTIMIZER = 'adam'
# FOLDS = [
#     (np.arange(1,12), np.arange(12,17), np.arange(18,23)),
#     (np.arange(6,17), np.arange(18,23), np.arange(1,6)),
#     (np.arange(12,23), np.arange(1,6), np.arange(6,12)),
#     (np.append(np.arange(1,6),np.arange(18,23)), np.arange(6,12), np.arange(12,17))
#     ]

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def read_image(path):
    """Reads image from path and returns image array"""
    img = skimage.io.imread(path, as_gray=True)

    return np.expand_dims(img, axis=2)

def extract_time(path):
    datetime_data = Image.open(path)._getexif()[36868]
    datetime_data = datetime_data[:-4]
    time = datetime.strptime(datetime_data, '%Y:%m:%d %H:%M:%S')
    return time

def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
      image = cv2.resize(image.transpose(1, 2, 0), resize)
      image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined

def postprocessing(mask):
    '''This function enabels post processing based on binary mask.
    This consists of selecting the largest object (one_object) and a closing
    operation to fill the holes (closing)
    Input: binary mask
    Returns: Post processing mask (0,1)
    ''' 
    labels_mask = label(mask)                       
    regions = regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
      for rg in regions[1:]:
        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    mask = labels_mask
    
    if np.max(mask) == 255:
      mask = mask/255
    im_flood_fill = mask.copy()
    h, w = mask.shape[:2]
    overlay = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, overlay, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    mask_out = mask | im_flood_fill_inv
    mask_out = mask_out/255
    return mask_out


def build_unet_model(input_height, input_width, input_channels):
    # Define input layer
    inputs = tf.keras.layers.Input((input_height, input_width, input_channels))

    # Normalize input data
    normalized_inputs = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Down sampling path
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(normalized_inputs)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    pool5 = tf.keras.layers.MaxPooling2D((2, 2))(conv5)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool5)

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


unique_ids = np.unique(np.array([int(file.split('_')[0][1:]) for file in os.listdir(IMG_PATH)]))

print(unique_ids)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for i_fold, (train_index, test_index) in enumerate(kf.split(unique_ids)):
    test_ids = unique_ids[test_index]
    train_ids = unique_ids[train_index]

    # TODO: change back to test_ids!

    # load trained model    
    model = build_rd_unet(input_height=IMG_HEIGHT, input_width=IMG_WIDTH, input_channels=IMG_CHANNELS)
    model.load_weights(OUTPUT_FOLDER + '/complete_model_' + str(i_fold+1) + '.h5')
    
    for folder in os.listdir(IMG_TIMELAPSE_PATH):
        OUTPUT_IMAGE_FOLDER_TEMP = OUTPUT_IMAGE_FOLDER + '/' + folder + '/'
        os.makedirs(OUTPUT_IMAGE_FOLDER_TEMP, exist_ok=True)

        if not os.path.isdir(IMG_TIMELAPSE_PATH + folder):
            continue
        if int(folder[1:]) not in test_ids: # change back!
            continue

        test_fn = sorted([file for file in os.listdir(IMG_PATH) if file.split('_')[0] == folder])
        test_fn_timelapse = sorted([file for file in os.listdir(IMG_TIMELAPSE_PATH + folder)])

        # Images and timepoints with available masks:
        test_images_annotated = np.array([read_image(IMG_PATH + file) for file in test_fn])
        test_masks_annotated = np.array([read_image(MASK_PATH + file.replace('.JPG', '.tif')) for file in test_fn], dtype=bool)
        test_images_annotated_timepoints = np.array([extract_time(IMG_PATH + file) for file in test_fn])
        test_images_annotated_timepoints = np.array([(time-test_images_annotated_timepoints[0]).total_seconds() / 60 /60 for time in test_images_annotated_timepoints])

        # Full timelapse data:
        test_images_timelapse = np.array([read_image(IMG_TIMELAPSE_PATH + folder + '/' + file) for file in test_fn_timelapse])
        test_images_timepoints = np.array([extract_time(IMG_TIMELAPSE_PATH + folder + '/' + file) for file in test_fn_timelapse])
        test_images_timepoints = np.array([(time-test_images_timepoints[0]).total_seconds() / 60 /60 for time in test_images_timepoints])
        
        # Predictions:
        test_predictions = model.predict(test_images_timelapse, batch_size=4)
        test_predictions_thresholded = test_predictions > 0.5

        # Post-process the predictions:
        test_predictions_thresholded = np.array([postprocessing(np.squeeze(img)) for img in test_predictions_thresholded])

        # Measure area of the crossection:
        test_images_areas = np.array([np.sum(img) for img in test_predictions_thresholded]) * IMG_PIXEL_SIZE
        test_images_gt_areas = np.array([np.sum(img) for img in test_masks_annotated]) * IMG_PIXEL_SIZE

        plt.figure()
        plt.title(f"Growth curve for ID {folder}")
        plt.plot(test_images_timepoints, test_images_areas, linestyle='dotted', marker='o', color='orange', label='Prediction')
        plt.plot(test_images_annotated_timepoints, test_images_gt_areas, linestyle='', marker='o', color='blue', label='Ground truth')
        plt.xlabel('time [h]')
        plt.ylabel(r'Area of the crossection $\mu$m$^2$')
        plt.legend()
        plt.savefig(OUTPUT_CURVE_FOLDER + f"Growth_curve_{folder}" + '.jpg')
        plt.close()

        for filename, image, segmentation in zip(test_fn_timelapse, test_images_timelapse, test_predictions_thresholded):
            se = disk(3)
            mask_dil = dilation(np.squeeze(segmentation), se)
            mask_ero = erosion(np.squeeze(segmentation), se)
            mask_border = np.logical_xor(mask_dil,mask_ero)*150
            image_3 = cv2.merge((image, image, image))
            image_with_mask = overlay(image_3, mask_border, color = (0,0,255), alpha=0.33)

            image_with_pred = Image.fromarray((image_with_mask))
            image_with_pred.save(OUTPUT_IMAGE_FOLDER_TEMP + filename.replace('.JPG', '.jpg'))

        pd.DataFrame({'time': test_images_timepoints, 'area': test_images_areas}).to_csv(OUTPUT_CURVE_FOLDER + f"Growth_curve_{folder}" + '.csv', index=False)
