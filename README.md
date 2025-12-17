# Blastocyst Segmentation (RD-U-Net)

A Deep Learning repository for the cross-sectional area segmentation of human blastocysts using a **Residual Dilated U-Net (RD-U-Net)**. This project includes custom data loading, specific loss functions (Weighted BCE + Dice), and morphological post-processing.

The algorithm is designed to segment the blastocyst structure from microscope images, facilitating automated analysis for embryology and IVF research.

## 📂 Repository Structure

```text
blastocyst-segmentation/
├── src/
│   ├── dataset.py       # DataGenerator with Albumentations
│   ├── loss.py          # Custom Weighted BCE + Dice Loss
│   ├── model.py         # RD-U-Net Architecture definition
│   └── utils.py         # Post-processing (hole filling, largest component)
├── train.py             # Training script
├── inference.py         # Inference script for new images
├── test_installation.py # Script to verify setup
├── requirements.txt     # Python dependencies
└── weights/             # Store your trained .h5 models here
```

