import argparse
import yaml
import os
import datetime

import numpy as np
import pandas as pd

from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import LearningRateScheduler

from dataset.datagenerator import DataGenerator
from utils.loss_functions import dice_loss, weighted_bce_dice_loss
from networks.model import build_unet, build_rd_unet


def process_experiment_file(file_path):
    """ Load the experiment file and return the data as a dictionary.
    Parameters
    ----------
    file_path : str
        Path to the .yaml file containing the experiment and dataset information
    """

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.1)


def main(): 
    parser = argparse.ArgumentParser(
        description='Train the best model on the dataset and save the results in the output folder.'
    )
    parser.add_argument("--experiment_file", type=str, required=True, help="Path to the .yaml file containing the experiment and dataset information")
    args = parser.parse_args()

    # Load the experiment file
    experiment_file = args.experiment_file
    experiment = process_experiment_file(experiment_file)

    output_folder = experiment['exp_dir']

    # Load the results CSV file
    file_path = os.path.join(os.path.dirname(experiment_file), "experiments.csv")
    results = pd.read_csv(file_path)

    # Select the best model and training parameters
    best_model = results.loc[results['Average_jaccardscore'].idxmax()]

    if best_model['loss'] not in ["binary_crossentropy", "binary_focal_crossentropy"]:
        loss = dice_loss if best_model['loss'] == "dice" else weighted_bce_dice_loss
    else:
        loss = best_model['loss']

    model = eval(best_model['model'])((800, 800, 1), normalization=best_model['normalization'])
    model.compile(optimizer=best_model['optimizer'], loss=loss, metrics=['accuracy', BinaryIoU(name='binary_io_u')])
 
    # Load the training data
    train_fn = sorted([file for file in os.listdir(experiment['img_dir'])])
    
    # Initialize datagenerator
    train_datagen = DataGenerator(list_IDs=train_fn,
                                  img_path=experiment['img_dir'],
                                  mask_path=experiment['mask_dir'],
                                  batch_size=best_model['batch_size'],
                                  dim=(800,800),
                                  n_channels=1,
                                  augmentation=['augmentation'])
    
    results = model.fit(train_datagen, epochs=experiment['n_epochs'], callbacks=[LearningRateScheduler(scheduler)])

    date = datetime.datetime.now().strftime("%Y%m%d")
    model.save(os.path.join(output_folder, f"Blast_segmentation_model_{date}.h5"))


if __name__ == "__main__":
    main()