import argparse
import yaml
import os
import csv 

import numpy as np
import pandas as pd
import wandb

from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, f1_score

from dataset.datagenerator import ClassificationDataGenerator
# from utils.loss_functions import dice_loss, weighted_bce_dice_loss
from networks.model import build_resnet50, transfer_model, trainable_model, model_rad, small_cnn, rdunet_features
from utils.postprocessing import postprocessing
from utils.metrics import specificity_score, save_loss_curve
from wandb.keras import WandbMetricsLogger

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
    # combined_dict_df['model'] = combined_dict_df['model'].values[0].__name__
    if callable(combined_dict_df['loss'].values[0]):
        combined_dict_df['loss'] = combined_dict_df['loss'].values[0].__name__

    # Append the combined dictionary to the DataFrame
    df = pd.concat([df, combined_dict_df], ignore_index=True)

    # Write the DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

# def scheduler(epoch, lr):
#     if (epoch+1) % 20 == 0:
#         return lr * 0.1
#     else:
#         return lr 


def main(): 
    parser = argparse.ArgumentParser(
        description='Train a model on a dataset'
    )
    parser.add_argument("--experiment_file", type=str, required=True, help="Path to the .yaml file containing the experiment and dataset information")
    parser.add_argument("--model", type=str, required=False, choices=["resnet50", "xception", "vgg16", "densenet121"] , help="Model to use for training")
    parser.add_argument("--optimizer", type=str, required=False, choices=["adam", "sgd"], help="Optimizer to use for training")
    parser.add_argument("--lr", type=float, required=False, help="Initial learning rate", default=0.001)
    parser.add_argument("--loss", type=str, required=False, choices=["binary_crossentropy", "binary_focal_crossentropy"], help="Loss function to use for training")
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, help="Flag to use data augmentation for training")
    parser.add_argument("--oversampling", action=argparse.BooleanOptionalAction, help="Flag to use data oversampling for training")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size for training")
    parser.add_argument("--expansion", type=str, required=False, help="Expansion features for training")

    args = parser.parse_args()

    # Load the experiment file
    experiment = process_experiment_file(args.experiment_file)

    # Set up additional experiment parameters
    experiment['model'] = args.model
    experiment['optimizer'] = args.optimizer
    experiment['lr'] = args.lr
    experiment['loss'] = args.loss
    experiment['augmentation'] = args.augmentation
    experiment['oversampling'] = args.oversampling
    experiment['batch_size'] = args.batch_size
    if args.expansion:
        experiment['expansion'] = True
    else:
        experiment['expansion'] = False

    # Make a folder for the experiment
    experiment_id = get_experiment_id(args.experiment_file)
    experiment['ID'] = experiment_id
    experiment_folder = os.path.join(experiment['exp_dir'], f"experiment_{experiment_id}")
    os.makedirs(experiment_folder, exist_ok=True)

    experiment_results = {}
    
    # Get the data for the selected fold
    for fold in range(experiment['n_folds']):
        print(f"[Experiment {experiment_id}] Training model for fold {fold+1}")

        # train_ids, val_ids, test_ids = experiment[f"Fold {fold+1}"]["Train set"], experiment[f"Fold {fold+1}"]["Validation set"], experiment[f"Fold {fold+1}"]["Test set"]
        train_ids, test_ids = experiment[f"Fold {fold+1}"]["Train set"], experiment[f"Fold {fold+1}"]["Test set"]

        # oversampling:
        if experiment['oversampling']:
            train_ids = train_ids * 20
 
        # Initialize datageneratorsimg_cropped
        train_datagen = ClassificationDataGenerator(list_IDs=train_ids,
                                      img_path=experiment['img_dir'],
                                      label_df=pd.read_csv(experiment['label_file']).set_index('ID'),
                                      batch_size=experiment['batch_size'],
                                      dim=(800, 800),
                                      n_channels=1,
                                      augmentation=experiment['augmentation'],
                                      mask_path=experiment['img_dir']+'/masks/',
                                      mode=3,
                                      feature_df=args.expansion)
        
        test_datagen = ClassificationDataGenerator(list_IDs=test_ids,
                                        img_path=experiment['img_dir'],
                                        label_df=pd.read_csv(experiment['label_file']).set_index('ID'),
                                        batch_size=8,
                                        dim=(800, 800),
                                        n_channels=1,
                                        shuffle=False,
                                        augmentation=False,
                                        mask_path=experiment['img_dir']+'/masks/',
                                        mode=3,
                                        feature_df=args.expansion)

        model = transfer_model(input_shape=(800, 800, 1), expansion=experiment['expansion'], base_model=experiment['model'], finetune=False)
        # model = trainable_model(input_shape=(800, 800, 1), expansion=experiment['expansion'], base_model=experiment['model'])
        # model = model_rad(input_shape=(800, 800, 1))

        # X, y = train_datagen.__getitem__(1)
        # plt.imshow(X[0][0].reshape(800, 800), cmap='gray')
        # print(X[1][0])
        # plt.show()
        # plt.imshow(X[1].reshape(800, 800), cmap='gray')
        # plt.show()
        # plt.imshow(X[2].reshape(800, 800), cmap='gray')
        # plt.show()

        # model = small_cnn(input_shape=(800, 800, 1), expansion=False)
        # model = rdunet_features()

        model.compile(optimizer=Adam(learning_rate=experiment['lr']), loss=experiment['loss'], metrics=['accuracy', AUC(name='auc')])

        # lr scheduler
        # lr_callback = LearningRateScheduler(scheduler)

        # early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=15, min_lr=0.0001)

        # init wandb run
        run = wandb.init(
            project="Blastocyst-Prediction",
            config=experiment,
            name=f'Classification_{experiment["ID"]}_fold_{fold+1}_{experiment["model"]}_expansion_{experiment["expansion"]}_oversampling_{experiment["oversampling"]}_lr_{experiment["lr"]}',
        )

        results = model.fit(train_datagen, validation_data=test_datagen, epochs=experiment['n_epochs'], callbacks=[reduce_lr, early_stopping, WandbMetricsLogger()])
        model.save(os.path.join(experiment_folder, f"model_fold_{fold+1}.h5"))

        # # Save loss curve
        # save_loss_curve(experiment_folder, results, fold)

        # Test the model and append results to csv file
       
        predictions, predictions_prob, labels = [], [], []

        for test_images, label in test_datagen:
            preds_test = model.predict(test_images)

            preds_test_th = preds_test > 0.5
            predictions.extend(preds_test_th)
            predictions_prob.extend(preds_test)
            labels.extend(label)

        # Calculate metrics            
        experiment_results[f"Fold{fold+1}_accuracyscore"] = accuracy_score(labels, predictions)
        experiment_results[f"Fold{fold+1}_roc_aucscore"] = roc_auc_score(labels, predictions_prob)
        experiment_results[f"Fold{fold+1}_sensitivityscore"] = recall_score(labels, predictions)
        experiment_results[f"Fold{fold+1}_specificityscore"] = specificity_score(labels, predictions)
        experiment_results[f"Fold{fold+1}_precisionscore"] = precision_score(labels, predictions)
        experiment_results[f"Fold{fold+1}_f1score"] = f1_score(labels, predictions)
        
        run.finish()

    # Calculate averages of the metrics
    experiment_results['Average_accuracyscore'] = np.mean([experiment_results[f"Fold{fold+1}_accuracyscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_roc_aucscore'] = np.mean([experiment_results[f"Fold{fold+1}_roc_aucscore"] for fold in range(experiment['n_folds'])])   
    experiment_results['Average_sensitivityscore'] = np.mean([experiment_results[f"Fold{fold+1}_sensitivityscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_specificityscore'] = np.mean([experiment_results[f"Fold{fold+1}_specificityscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_precisionscore'] = np.mean([experiment_results[f"Fold{fold+1}_precisionscore"] for fold in range(experiment['n_folds'])])
    experiment_results['Average_f1score'] = np.mean([experiment_results[f"Fold{fold+1}_f1score"] for fold in range(experiment['n_folds'])])


    # Append results to CSV
    append_to_csv(args.experiment_file, 
                  {key: value for key, value in experiment.items() if key in ['ID', 'img_dir', 'mask_dir', 'exp_dir', 'n_folds', 'n_epochs', 'model', 'optimizer', 'lr', 'loss', 'augmentation', 'oversampling', 'expansion', 'batch_size']},
                  experiment_results)

if __name__ == "__main__":
    main()
