import argparse
import os
import yaml
import csv

import numpy as np

# Ensure reproducibility
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser(
        description='Prepare a folder with dataset information for for running multiple experiments with the same dataset.'
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing the images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the folder containing the masks")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the folder containing the experiments output")
    parser.add_argument("--n_folds", type=int, default=4, help="Number of folds for cross-validation")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs for training")

    args = parser.parse_args()

    # Check if experiment_dir exists else create it
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Initialize the output
    output = {
        "img_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "exp_dir": args.experiment_dir,
        "n_folds": args.n_folds,
        "n_epochs": args.n_epochs,
    }

    files = os.listdir(args.mask_dir)
    ids = np.unique([int(file.split("_")[0][1:]) for file in files])

    # Shuffle the IDs randomly
    np.random.shuffle(ids)
    print(len(ids))

    # Split the IDs into 4 folds
    folds = np.array_split(ids, output['n_folds'])

    # Initialize a set to store the IDs that have been used for internal validation
    # used_ids = set()

    # Loop over the folds
    for i in range(output['n_folds']):
        # Select the ith fold as the test set
        test_set = folds[i]
        
        # Concatenate the remaining folds as the training set
        train_set = np.concatenate([folds[j] for j in range(output['n_folds']) if j != i])
        
        # Select 20% of the training set as the internal validation set
        # Make sure the IDs are not in the used_ids set
        # valid_set = np.random.choice([id for id in train_set if id not in used_ids], size=int(len(train_set) * 0.2), replace=False)
        
        # Add the IDs in the validation set to the used_ids set
        # used_ids.update(valid_set)
        
        # Store the test, train, and validation sets for the ith fold in the output dictionary
        # output[f"Fold {i+1}"] = {"Test set": test_set.tolist(), "Train set": train_set.tolist(), "Validation set": valid_set.tolist()}
        output[f"Fold {i+1}"] = {"Test set": test_set.tolist(), "Train set": train_set.tolist()}

    # Open a file for writing the output data in YAML format
    with open(os.path.join(args.experiment_dir, "output.yaml"), "w") as yaml_file:
        # Dump the output data to the file using the yaml module
        yaml.dump(output, yaml_file, default_flow_style=False, sort_keys=False)

    # Prepare the CSV file for the experiments
    
    header_dict = ["ID"] + [key for key in output.keys() if "Fold" not in key] + ["model", "optimizer", "loss", "augmentation", "normalization", "batch_size"]

    # Specify the CSV file name
    csv_file_name = 'experiments.csv'

    # Check if the CSV file already exists
    try:
        with open(os.path.join(args.experiment_dir, csv_file_name), 'x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header_dict)
            writer.writeheader()
    except FileExistsError:
        pass

if __name__ == "__main__":
    main()