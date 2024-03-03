import argparse
import yaml
import os
import csv 

import numpy as np
import pandas as pd

from operator import itemgetter
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score, jaccard_score, recall_score, precision_score

from dataset.datagenerator import DataGenerator
from utils.loss_functions import dice_loss, weighted_bce_dice_loss
from networks.model import build_unet, build_rd_unet
from utils.postprocessing import postprocessing
from utils.metrics import specificity_score, save_loss_curve

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

    # Load the existing CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Merge the experiment and results dictionaries into a single dictionary
    combined_dict = {**experiment_dict, **results_dict}
    combined_dict_df = pd.DataFrame(combined_dict, index=[0])
    combined_dict_df['model'] = combined_dict_df['model'].values[0].__name__
    if callable(combined_dict_df['loss'].values[0]):
        combined_dict_df['loss'] = combined_dict_df['loss'].values[0].__name__

    # Append the combined dictionary to the DataFrame
    df = pd.concat([df, combined_dict_df], ignore_index=True)

    # Write the DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.1)


def main(): 
    parser = argparse.ArgumentParser(
        description='Train a model on a dataset'
    )
    parser.add_argument("--experiment_file", type=str, required=True, help="Path to the .yaml file containing the experiment and dataset information")
    parser.add_argument("--model", type=str, required=False, choices=["unet", "rd_unet"] , help="Model to use for training")
    parser.add_argument("--optimizer", type=str, required=False, choices=["adam", "sgd"], help="Optimizer to use for training")
    parser.add_argument("--loss", type=str, required=False, choices=["binary_crossentropy", "binary_focal_crossentropy", "dice", "dice_bce"], help="Loss function to use for training")
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, help="Flag to use data augmentation for training")
    parser.add_argument("--normalization", type=str, required=False, choices=["min_max", "batchnorm"], help="Normalization method for input data")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training")

    args = parser.parse_args()

    # Load the experiment file
    experiment = process_experiment_file(args.experiment_file)

    # Set up additional experiment parameters
    experiment['model'] = build_unet if args.model == "unet" else build_rd_unet
    experiment['optimizer'] = args.optimizer
    if args.loss not in ["binary_crossentropy", "binary_focal_crossentropy"]:
        experiment['loss'] = dice_loss if args.loss == "dice" else weighted_bce_dice_loss
    else:
        experiment['loss'] = args.loss
    experiment['augmentation'] = args.augmentation
    experiment['batch_size'] = args.batch_size
    experiment['normalization'] = args.normalization

    # Make a folder for the experiment
    experiment_id = get_experiment_id(args.experiment_file)
    experiment['ID'] = experiment_id
    experiment_folder = os.path.join(experiment['exp_dir'], f"experiment_{experiment_id}")
    os.makedirs(experiment_folder)

    experiment_results = {}

    # Get the data for the selected fold
    for fold in range(experiment['n_folds']):
        print(f"[Experiment {experiment_id}] Training model for fold {fold+1}")

        # train_ids, val_ids, test_ids = experiment[f"Fold {fold+1}"]["Train set"], experiment[f"Fold {fold+1}"]["Validation set"], experiment[f"Fold {fold+1}"]["Test set"]
        train_ids, test_ids = experiment[f"Fold {fold+1}"]["Train set"], experiment[f"Fold {fold+1}"]["Test set"]

        # Identify training, validation and test images if the filename contains the id of the patient:
        train_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in train_ids])
        # val_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in val_ids])
        test_fn = sorted([file for file in os.listdir(experiment['img_dir']) if int(file.split('_')[0][1:]) in test_ids])

        # Initialize datagenerators
        train_datagen = DataGenerator(list_IDs=train_fn,
                                      img_path=experiment['img_dir'],
                                      mask_path=experiment['mask_dir'],
                                      batch_size=experiment['batch_size'],
                                      dim=(800,800),
                                      n_channels=1,
                                      augmentation=['augmentation'])
        
        # validation_datagen = DataGenerator(list_IDs=val_fn,
        #                                    img_path=experiment['img_dir'],
        #                                    mask_path=experiment['mask_dir'],
        #                                    batch_size=experiment['batch_size'],
        #                                    dim=(800,800),
        #                                    n_channels=1,
        #                                    augmentation=False)
        
        test_datagen = DataGenerator(list_IDs=test_fn,
                                        img_path=experiment['img_dir'],
                                        mask_path=experiment['mask_dir'],
                                        batch_size=8,
                                        dim=(800,800),
                                        n_channels=1,
                                        shuffle=False,
                                        augmentation=False)
        
        model = experiment['model'](input_shape=(800, 800, 1), normalization=args.normalization, print_summary=False)
        model.compile(optimizer=experiment['optimizer'], loss=experiment['loss'], metrics=['accuracy', BinaryIoU(name='binary_io_u')])

        # lr scheduler
        lr_callback = LearningRateScheduler(scheduler)

        results = model.fit(train_datagen, validation_data=test_datagen, epochs=experiment['n_epochs'], callbacks=[lr_callback])
        model.save(os.path.join(experiment_folder, f"model_fold_{fold+1}.h5"))

        # Save loss curve
        save_loss_curve(experiment_folder, results, fold)

        # Test the model and append results to csv file
       
        dice_scores = []
        jaccard_scores = []
        sensitivity_scores = []
        specificity_scores = []
        precision_scores = []

        for test_images, test_masks in test_datagen:
            preds_test = model.predict(test_images)
            preds_test_t = (preds_test > 0.5).astype(np.uint8)

            # Postprocess batch
            processed_preds_test_t = postprocessing(preds_test_t)

            # Calculate metrics            
            dice_scores.append(f1_score(test_masks.flatten(), processed_preds_test_t.flatten()))
            jaccard_scores.append(jaccard_score(test_masks.flatten(), processed_preds_test_t.flatten()))
            sensitivity_scores.append(recall_score(test_masks.flatten(), processed_preds_test_t.flatten()))
            specificity_scores.append(specificity_score(test_masks.flatten(), processed_preds_test_t.flatten()))
            precision_scores.append(precision_score(test_masks.flatten(), processed_preds_test_t.flatten()))



        experiment_results[f"Fold{fold+1}_dicescore"] = np.mean(dice_scores)
        experiment_results[f"Fold{fold+1}_jaccardscore"] = np.mean(jaccard_scores)
        experiment_results[f"Fold{fold+1}_sensitivityscore"] = np.mean(sensitivity_scores)
        experiment_results[f"Fold{fold+1}_specificityscore"] = np.mean(specificity_scores)
        experiment_results[f"Fold{fold+1}_precisionscore"] = np.mean(precision_scores)

    # Calculate averages of the metrics
    experiment_results['Average_dicescore'] = np.mean([experiment_results[f"Fold{fold+1}_dicescore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_jaccardscore'] = np.mean([experiment_results[f"Fold{fold+1}_jaccardscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_sensitivityscore'] = np.mean([experiment_results[f"Fold{fold+1}_sensitivityscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_specificityscore'] = np.mean([experiment_results[f"Fold{fold+1}_specificityscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_precisionscore'] = np.mean([experiment_results[f"Fold{fold+1}_precisionscore"] for fold in range(experiment['n_folds'])])

    # Append results to CSV
    append_to_csv(args.experiment_file, 
                  {key: value for key, value in experiment.items() if key in ['ID', 'img_dir', 'mask_dir', 'exp_dir', 'n_folds', 'n_epochs', 'model', 'optimizer', 'loss', 'augmentation', 'normalization', 'batch_size']}, 
                  experiment_results)

        
if __name__ == "__main__":
    main()
