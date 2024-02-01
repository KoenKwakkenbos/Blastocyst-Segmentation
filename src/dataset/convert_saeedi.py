import os
from PIL import Image, ImageDraw, ImageChops

DATASET_TE_MASKS_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Data\BlastsOnline\GT_ZP"
OUTPUT_BINARY_MASK = r"C:\Users\koenk\Documents\Master_Thesis\Data\SaeediBinary\Masks"

def hole_fill(image_path, output_path):
    # Open the BMP image
    img = Image.open(image_path)

    # Convert the image to RGB (if it's not already)
    img = img.convert("RGB")

    # Extract the binary mask from the blue channel (assuming it's a 24-bit BMP with R, G, B channels)
    binary_mask = img.split()[2]

    # Create a copy of the binary mask
    filled_mask = binary_mask.copy()

    # Perform hole-filling using flood-fill
    ImageDraw.floodfill(filled_mask, (0, 0), 255)

    # invert filled_mask:
    result_img = ImageChops.invert(filled_mask)

    # Save the result
    result_img.save(output_path)

if __name__ == "__main__":
    input_files = [f for f in os.listdir(DATASET_TE_MASKS_PATH) if f.endswith('.bmp')]

    for file in input_files:
        input_image_path = os.path.join(DATASET_TE_MASKS_PATH, file)
        output_image_path = os.path.join(OUTPUT_BINARY_MASK, file.replace('.bmp', '_binary.bmp'))
        hole_fill(input_image_path, output_image_path)
