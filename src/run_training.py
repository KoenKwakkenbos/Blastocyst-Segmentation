import argparse
import yaml
import os

import numpy as np

from tensorflow.keras.metrics import BinaryIoU
from sklearn.metrics import f1_score, jaccard_score

from dataset.datagenerator import DataGenerator
from utils.loss_functions import dice_loss, weighted_bce_dice_loss
from networks.model import build_unet, build_rd_unet
from utils.postprocessing import postprocessing

import matplotlib.pyplot as plt

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

def get_experiment_id(exp_file):
    """ Get the next experiment ID based on the existing folders in the experiment directory.
    Parameters
    ----------
    exp_file : str
        Path to the experiment file
    """

    folder_path = os.path.dirname(os.path.abspath(exp_file))
    folders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
    ids = [int(folder[-3:]) for folder in folders if folder[-3:].isdigit()]
    if ids:
        max_id = max(ids)
        new_id = str(max_id + 1).zfill(3)
    else:
        new_id = "001"
    return new_id

def append_to_csv(experiment_file, experiment_dict, results_dict):
    """ Append the experiment and results to a CSV file.
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    experiment_dict : dict
        Dictionary containing the experiment parameters
    results_dict : dict
        Dictionary containing the results of the experiment
    """

    file_path = os.path.join(os.path.dirname(experiment_file), "experiments.csv")

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Append the experiment and results to the file
    with open(file_path, "a") as file:
        experiment_values = ",".join([str(val) for val in experiment_dict.values()])
        results_values = ",".join([str(val) for val in results_dict.values()])
        file.write(experiment_values + "," + results_values + "\n")


def main(): 
    parser = argparse.ArgumentParser(
        description='Train a model on a dataset'
    )
    parser.add_argument("--experiment_file", type=str, required=True, help="Path to the .yaml file containing the experiment and dataset information")
    parser.add_argument("--model", type=str, required=False, choices=["unet", "rd_unet"] , help="Model to use for training")
    parser.add_argument("--optimizer", type=str, required=False, choices=["adam", "sgd"], help="Optimizer to use for training")
    parser.add_argument("--loss", type=str, required=False, choices=["binary_crossentropy", "binary_focal_crossentropy", "dice", "dice_bce"], help="Loss function to use for training")
    parser.add_argument("--augmentation", type=bool, required=False, help="Whether to use data augmentation for training")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training")

    args = parser.parse_args()

    # Load the experiment file
    experiment = process_experiment_file(args.experiment_file)

    # Set up additional experiment parameters
    experiment['model_func'] = build_unet if args.model == "unet" else build_rd_unet
    experiment['optimizer'] = args.optimizer
    if args.loss not in ["binary_crossentropy", "binary_focal_crossentropy"]:
        experiment['loss'] = dice_loss if args.loss == "dice" else weighted_bce_dice_loss
    experiment['augmentation'] = args.augmentation
    experiment['batch_size'] = args.batch_size

    # Make a folder for the experiment
    experiment_id = get_experiment_id(args.experiment_file)
    experiment_folder = os.path.join(experiment['exp_dir'], f"experiment_{experiment_id}")
    os.makedirs(experiment_folder)

    experiment_results = {}

    # Get the data for the selected fold
    for fold in range(experiment['n_folds']):
        train_ids, val_ids, test_ids = experiment[f"Fold {fold+1}"]["Train set"], experiment[f"Fold {fold+1}"]["Validation set"], experiment[f"Fold {fold+1}"]["Test set"]

        # Identify training, validation and test images if the filename contains the id of the patient:
        train_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in train_ids])
        val_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in val_ids])
        test_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in test_ids])

        # Initialize datagenerators
        train_datagen = DataGenerator(list_IDs=train_fn,
                                      img_path=experiment['img_dir'],
                                      mask_path=experiment['mask_dir'],
                                      batch_size=experiment['batch_size'],
                                      dim=(800,800),
                                      n_channels=1)
        
        validation_datagen = DataGenerator(list_IDs=val_fn,
                                           img_path=experiment['img_dir'],
                                           mask_path=experiment['mask_dir'],
                                           batch_size=experiment['batch_size'],
                                           dim=(800,800),
                                           n_channels=1,
                                           train=False)
        
        test_datagen = DataGenerator(list_IDs=test_fn,
                                        img_path=experiment['img_dir'],
                                        mask_path=experiment['mask_dir'],
                                        batch_size=8,
                                        dim=(800,800),
                                        n_channels=1,
                                        shuffle=False,
                                        train=False)
        
        model = experiment['model_func'](input_shape=(800, 800, 1))
        model.compile(optimizer=args.optimizer, loss=args.loss, metrics=['accuracy', BinaryIoU()])

        results = model.fit(train_datagen, validation_data=validation_datagen, epochs=experiment['n_epochs'])
        # model.save(os.path.join(experiment_folder, f"model_fold_{fold+1}.h5"))

        # # Test the model and append results to csv file
       
        f1_scores = []
        jaccard_scores = []

        for test_images, test_masks in test_datagen:
            preds_test = model.predict(test_images)
            preds_test_t = (preds_test > 0.5).astype(np.uint8)

            processed_preds_test_t = postprocessing(preds_test_t)

            f1_scores.append(f1_score(test_masks.flatten(), processed_preds_test_t.flatten()))
            jaccard_scores.append(jaccard_score(test_masks.flatten(), processed_preds_test_t.flatten()))


        experiment_results[f"Fold{fold+1}_f1score"] = np.mean(f1_scores)
        experiment_results[f"Fold{fold+1}_jaccardscore"] = np.mean(jaccard_scores)

    # Save results of experiment
    append_to_csv(args.experiment_file, experiment, experiment_results)

        
if __name__ == "__main__":
    main()
