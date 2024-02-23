
import numpy as np

from skimage.morphology import disk, erosion, dilation
from skimage.measure import regionprops, label
from cv2 import morphologyEx, MORPH_CLOSE, floodFill, bitwise_not


def postprocessing(mask):
    """Postprocessing function to clean up binary masks.

    Parameters
    ----------
    mask : np.ndarray
        Binary masks of shape (height, width, num_masks) or (height, width).
    """

    mask = np.squeeze(mask)

    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=0)
    
    # Assume mask is a numpy array of shape (h, w, n) where n is the number of masks
    n, h, w, = mask.shape
    labels_mask = np.zeros_like(mask) # Initialize labels_mask with the same shape as mask
    mask_out = np.zeros_like(mask) # Initialize mask_out with the same shape as mask

    for i in range(n): # Loop over each mask
        labels_mask[i,] = label(mask[i,]) # Label each mask separately
        regions = regionprops(labels_mask[i,]) # Get the regions of each mask
        regions.sort(key=lambda x: x.area, reverse=True) # Sort the regions by area in descending order
        if len(regions) > 1: # If there are more than one region
            for rg in regions[1:]: # Loop over the smaller regions
                labels_mask[i, rg.coords[:,0], rg.coords[:,1]] = 0 
        labels_mask[i, labels_mask[i,] != 0] = 1
        mask[i,] = labels_mask[i,] # Update the mask with the labels
        
        if np.max(mask[i,]) == 255:
            mask[i,] = mask[i,] / 255
        im_flood_fill = mask[i,].copy()
        overlay = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8") # Convert the mask to uint8
        floodFill(im_flood_fill, overlay, (0, 0), 255) # Perform flood fill from the top-left corner
        im_flood_fill_inv = bitwise_not(im_flood_fill) # Invert the flood filled image
        mask_out[i,] = mask[i,] | im_flood_fill_inv # Combine the mask and the inverted image with bitwise OR
        mask_out[i,] = mask_out[i,] / 255 # Normalize the output mask to 0 or 1
    return mask_out # Return the output mask
