"""
Binary Thresholding
By Carlos Santiago Bañón

binary_thresholding.py
An implementation of the binary image thresholding algorithm.
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import sys
from skimage.color import rgb2gray


def binary_threshold(img, threshold):
    """
    Perform binary thresholding.

    :img: the original grayscale image.
    :threshold: the threshold value.
    """

    # Flatten the image.
    flat = np.ndarray.flatten(img)

    # Perform the threshold.
    for i in range(flat.shape[0]):

        if (flat[i] * 255) > threshold:
            flat[i] = 255
        else:
            flat[i] = 0

    # Reshape the image.
    thresh_img = np.reshape(flat, (img.shape[0], img.shape[1]))
  
    return thresh_img

# Load the image.
image_path = str(sys.argv[1])
img = mpimg.imread(image_path)

# Show the image.
plt.imshow(img, cmap='gray')
plt.show()

# Convert to a grayscale image.
grayscale = rgb2gray(img)

# Show the grayscale image.
plt.imshow(grayscale, cmap='gray')
plt.show()

# Get a histogram for the image.
histogram, bin_edges = np.histogram(grayscale, bins=256, range=(0, 1))

# Show the histogram.
plt.figure()
plt.xlabel("Value")
plt.ylabel("Number of Pixels")
plt.xlim([0.0, 255.0])
plt.plot((bin_edges[0:-1] * 255), histogram)
plt.show()

# Perform Otsu thresholding.
thresh_img = binary_threshold(grayscale, 185)

# Show the thresholded image.
plt.imshow(thresh_img, cmap='gray')
plt.show()