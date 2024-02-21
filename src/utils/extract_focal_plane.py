import os
import shutil

base_folder_path = r"C:\Users\koenk\Documents\Master_Thesis\Data\Raw_data\Expansion"
output_folder_path = r"C:\Users\koenk\Documents\Master_Thesis\Data\Processed_data\Expansion"

# Get the list of base folders
base_folders = [folder for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]

for base_folder in base_folders:
    # Create a new folder with the name of the base folder at the specified output path
    new_folder = os.path.join(output_folder_path, base_folder)
    os.makedirs(new_folder, exist_ok=True)

    # Get the path of the F0 subfolder
    f0_folder = os.path.join(base_folder_path, base_folder, "F0")

    # Copy all the files from F0 to the new folder
    for file_name in os.listdir(f0_folder):
        file_path = os.path.join(f0_folder, file_name)
        if os.path.isfile(file_path):
            shutil.copy2(file_path, new_folder)

    print(f"Copied images from {f0_folder} to {new_folder}")
