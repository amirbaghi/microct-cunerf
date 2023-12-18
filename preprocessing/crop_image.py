import cv2
import numpy as np
import os
import re
import math

# Directory where images are kept.
directory = "./preprocessing/images/"

# Default values to get max width and height of image
maxH = -math.inf
maxW = -math.inf

def rescaleImage(input_image_path, output_image_path):
    # Add padding to make all images the same size
    cropped_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    h, w = cropped_image.shape
    maxLen = max(maxH, maxW)

    # Calculate padding for height and width
    pad_h = maxLen - h
    pad_w = maxLen - w

    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad

    # Create a square image
    padded_image = cv2.copyMakeBorder(cropped_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite(output_image_path, padded_image)

def crop_image(input_image_path, output_image_path):
    # Read the image in 16-bit depth
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to 8-bit for finding contours
    image_8bit = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))

    # Apply thresholding to segment the image
    _, thresholded = cv2.threshold(image_8bit, 105, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the contour with the largest area (assuming it's the main object)
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the region of interest
    mask = np.zeros_like(image_8bit)
    cv2.drawContours(mask, [max_contour], 0, 255, thickness=cv2.FILLED)

    # Bitwise AND the original 16-bit image with the mask to get the cropped region
    mask_16bit = cv2.convertScaleAbs(mask, alpha=(65535.0/255.0))
    result = cv2.bitwise_and(image, image, mask=mask_16bit)

    # Find the bounding box of the cropped region
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image
    cropped_image = result[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)

    return (w,h)

# Crop all images in directory and pad them to the largest cropped image.
for filename in os.listdir(directory):
    # When re-running the script, remove previous padded images, to not repad padded images.
    if ("cropped_" in filename or "pp_" in filename):
        os.remove(directory + filename)
        continue
    input_path = directory + filename
    output_path = directory + "cropped_" + filename
    w,h = crop_image(input_path, output_path)
    maxH = max(maxH, h)
    maxW = max(maxW, w)

# Add padding to the cropped images..
for filename in os.listdir(directory):
    if ("cropped_" in filename):
        input_path = directory + filename
        output_path = directory + "pp_" + filename
        output_path = output_path.replace("cropped_", "" ) 
        rescaleImage(input_path, output_path)
        os.remove(directory + filename)