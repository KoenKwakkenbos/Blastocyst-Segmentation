import cv2
import os

import numpy as np
from skimage import io

def read_image(path):
    """Reads image from path and returns image array"""
    img = io.imread(path)

    return img[..., ::-1]
    # return img

def display_images(images, window_name='Blastocyst Timelapse'):
    is_auto_increment = False
    i = 0
    interval = 50

    def on_mouse_wheel(event, x, y, flags, param):
        nonlocal i
        nonlocal is_auto_increment
        if event == cv2.EVENT_MOUSEWHEEL:
            is_auto_increment = False
            # Scroll forward
            if flags > 0:
                i = min(i + 1, len(images) - 1)
            # Scroll backward
            else:
                i = max(i - 1, 0)
            cv2.imshow(window_name, images[i])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_wheel)

    while True:
        if is_auto_increment:
            
            if i + 1 >= len(images):
                is_auto_increment = False
            else:
                i = i + 1 # Increment the frame index

            cv2.imshow(window_name, images[i])  # Display the current image

            # Check for user input (spacebar, mousewheel, a/d keys)
            key = cv2.waitKey(interval) & 0xFF
            if key == ord(' '):  # Toggle auto incrementation
                is_auto_increment = not is_auto_increment
                interval = 50
            elif key == ord('q'):  # Exit
                break
            elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
                break
            elif key == ord('d'):  # Go forward
                i = min(i + 1, len(images) - 1)
                is_auto_increment = False
            elif key == ord('a'):  # Go backward
                i = max(i - 1, 0)
                is_auto_increment = False
            elif key == ord('.'):
                interval += 25 % 500
            elif key == ord(','):
                interval = max(interval - 25, 0)

        else:  # Manual mode
            cv2.imshow(window_name, images[i])  # Display the current image

            # Check for user input (spacebar, mousewheel, a/d keys)
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):  # Toggle auto incrementation
                is_auto_increment = not is_auto_increment
                if i == len(images) - 1:
                    i = 0
            elif key == ord('q'):  # Exit
                break
            elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
                break 
            elif key == ord('d'):  # Go forward
                i = min(i + 1, len(images) - 1)
            elif key == ord('a'):  # Go backward
                i = max(i - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\koenk\Documents\Master_Thesis\Programming\Debugging\experiment_012\Segmented_images\E38" #32
    image_files = sorted([file for file in os.listdir(INPUT_PATH) if file.endswith('.jpg')])
    images = [read_image(os.path.join(INPUT_PATH, file)) for file in image_files]

    display_images(images, window_name='Blastocyst Timelapse')

 