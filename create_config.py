import yaml

# Specify general parameters
data = {
    'general': {
        'image_path': 'value1',
        'mask_path': 'value2',
        'output_path': 'value3',
        'image_width': 800,
        'image_height': 800,
        'n_channels': 1,
        'batch_size': 8,
        'epochs': 100,
        'test': True,
        'n_folds': 4,
        'metrics': ['accuracy', 'binary_iou'],
        'random_seed': 42
    }
}

# Specify experiment parameters
data['experiments'] = {
    'model': ['rd_unet', 'unet'],
    'optimizer': 'adam',
    'loss_fn': 'binary_focal_crossentropy',
    'callbacks': ['lr_scheduler', 'early_stopping']
}

                                 
with open('./output.yaml', 'w') as file:
    yaml.dump(data, file)