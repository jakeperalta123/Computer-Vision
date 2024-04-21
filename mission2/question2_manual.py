import cv2
import numpy as np

# Read the image
image = cv2.imread('mission2B.png', cv2.IMREAD_GRAYSCALE)

# Compute gradients using Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Compute products of gradients
Ix2 = sobel_x ** 2
Iy2 = sobel_y ** 2
IxIy = sobel_x * sobel_y

# Define a window size for smoothing
window_size = 5
half_window = window_size // 2

# Compute corner response function R for each pixel
k = 0.04  # Empirical constant
R = np.zeros_like(image, dtype=np.float64)

for y in range(half_window, image.shape[0] - half_window):
    for x in range(half_window, image.shape[1] - half_window):
        # Compute sums of products of gradients in the window
        Sxx = np.sum(Ix2[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1])
        Syy = np.sum(Iy2[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1])
        Sxy = np.sum(IxIy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1])

        # Compute the determinant and trace of the matrix M
        det_M = (Sxx * Syy) - (Sxy ** 2)
        trace_M = Sxx + Syy

        # Compute the corner response function
        R[y, x] = det_M - k * (trace_M ** 2)

# Threshold the corner response function
threshold = 0.1 * np.max(R)
corners = np.zeros_like(image)
corners[R > threshold] = 255

# Display the original image and detected corners
cv2.imshow('Original Image', image)
cv2.imshow('Detected Corners', corners.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
