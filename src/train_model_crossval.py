# Imports
import os
import sys

import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, recall_score
from sklearn.model_selection import KFold

import src
from src.utils.loss_functions import dice_loss, weighted_bce_dice_loss
from dataset.datagenerator import DataGenerator
from src.networks.model import build_unet_model, UNet, build_rd_unet


# Experiment variables
EXP_NUM = 3
EXP_NAME = 'experiment_' + f"{EXP_NUM:03}"
TEST = True
IMG_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\images/"
MASK_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\masks/"
OUTPUT_PATH =  r"C:\Users\koenk\Documents\Master_Thesis\Programming\Debugging\DebuggingNew/"

# Image information
IMG_WIDTH = 800
IMG_HEIGHT = 800
IMG_CHANNELS = 1

# Model training
RANDOM_SEED = 1
BATCH_SIZE = 4
EPOCHS = 100
N_FOLDS = 4
LOSS_FN = dice_loss
OPTIMIZER = 'adam'

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=35, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

# Create output directory for experiment
# check if path exists and create if not:
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, EXP_NAME)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print('Created output folder:', OUTPUT_FOLDER)
else:
    if os.listdir(OUTPUT_FOLDER) == []:
        print(OUTPUT_FOLDER, 'already exists, but is empty -> running experiment.')
    else:
        print('Folder', OUTPUT_FOLDER, 'already exists, change experiment name')
        exit()

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.1)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def read_image(path):
    """Reads image from path and returns image array"""
    img = skimage.io.imread(path, as_gray=True)

    return np.expand_dims(img, axis=2)

def save_loss_curve(results, i_fold):
    epochs = range(len(results.history["val_loss"]))
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(epochs, results.history["accuracy"], label = 'training')
    ax1.plot(epochs, results.history["val_accuracy"], label = 'validation')
    ax1.set(xlabel = 'Epochs', ylabel ='accuracy')
    ax1.legend()

    ax2.plot(epochs, results.history["loss"], label = 'training')
    ax2.plot(epochs, results.history["val_loss"], label = 'validation')
    ax2.set(xlabel = 'Epochs',ylabel = 'Loss')
    ax2.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'val_acc_epochs' + str(i_fold+1) + '.png'))

unique_ids = np.unique(np.array([int(file.split('_')[0][1:]) for file in os.listdir(IMG_PATH)]))

print(unique_ids)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for i_fold, (train_index, test_index) in enumerate(kf.split(unique_ids)):
    test_ids = unique_ids[test_index]
    train_ids = unique_ids[train_index]

    val_ids = np.random.choice(train_ids, size=int(len(train_ids)*0.2), replace=False)
    train_ids = np.setdiff1d(train_ids, val_ids)

    print('Train ids:', train_ids)
    print('Validation ids:', val_ids)
    print('Test ids:', test_ids)

    print(f'Fold {i_fold+1}:')

    # Identify training, validation and test images if the filename contains the id of the patient:
    train_fn = sorted([file for file in os.listdir(IMG_PATH) if int(file.split('_')[0][1:]) in train_ids])
    val_fn = sorted([file for file in os.listdir(IMG_PATH) if int(file.split('_')[0][1:]) in val_ids])
    test_fn = sorted([file for file in os.listdir(IMG_PATH) if int(file.split('_')[0][1:]) in test_ids])

    # Initialize datagenerators    
    train_datagen = DataGenerator(list_IDs=train_fn, 
                                  img_path=IMG_PATH,
                                  mask_path=MASK_PATH,
                                  batch_size=BATCH_SIZE,
                                  dim=(800,800),
                                  n_channels=1)
    
    validation_datagen = DataGenerator(list_IDs=val_fn,
                                       img_path=IMG_PATH,
                                       mask_path=MASK_PATH,
                                       batch_size=BATCH_SIZE,
                                       dim=(800,800),
                                       n_channels=1,
                                       train=False)

    test_datagen = DataGenerator(list_IDs=test_fn,
                                 img_path=IMG_PATH,
                                 mask_path=MASK_PATH,
                                 batch_size=1,
                                 dim=(800,800),
                                 n_channels=1,
                                 shuffle=False,
                                 train=False)

    model = build_rd_unet(800, 800, 1)
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=['accuracy', tf.keras.metrics.BinaryIoU()])

    #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    results = model.fit(train_datagen, validation_data=validation_datagen, epochs=EPOCHS, callbacks=[lr_callback, es_callback])
    model.save(os.path.join(OUTPUT_FOLDER, 'complete_model_' + str(i_fold+1) + '.h5'))

    # Creating a figure of the accuracy and loss per epoch
    save_loss_curve(results, i_fold)

    if TEST:
        test_masks = np.array([read_image(MASK_PATH + file.replace('.JPG', '.tif')) for file in test_fn], dtype=bool)

        preds_test = model.predict(test_datagen)
        preds_test_t = (preds_test > 0.5).astype(np.uint8) 

        test_dice = np.array([f1_score(mask.flatten(), mask_t.flatten()) for (mask, mask_t) in zip(test_masks, preds_test_t)])
        test_jaccard = np.array([jaccard_score(mask.flatten(), mask_t.flatten()) for (mask, mask_t) in zip(test_masks, preds_test_t)])
        test_sensitivity = np.array([recall_score(mask.flatten(), mask_t.flatten()) for (mask, mask_t) in zip(test_masks, preds_test_t)])
        test_specificity = np.array([specificity_score(mask.flatten(), mask_t.flatten()) for (mask, mask_t) in zip(test_masks, preds_test_t)])

        print(f'Test dice score: {np.mean(test_dice):0.3f} +/- {np.std(test_dice):0.3f}')
        print(f'Test jaccard score: {np.mean(test_jaccard):0.3f} +/- {np.std(test_jaccard):0.3f}')
        print(f'Test sensitivity score: {np.mean(test_sensitivity):0.3f} +/- {np.std(test_sensitivity):0.3f}')
        print(f'Test specificity score: {np.mean(test_specificity):0.3f} +/- {np.std(test_specificity):0.3f}')

        pd.DataFrame({'dice': test_dice, 'jaccard': test_jaccard, 'sensitivity': test_sensitivity, 'specificity': test_specificity}).to_csv(os.path.join(OUTPUT_FOLDER, 'test_scores_' + str(i_fold+1) + '.csv'))
 