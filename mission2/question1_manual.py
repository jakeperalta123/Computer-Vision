import numpy as np
import cv2

# Read the image
image = cv2.imread('mission2A.png', cv2.IMREAD_GRAYSCALE)

# Gaussian blur
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

# Sobel operator for gradient calculation
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude and direction
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

# Non-maximum suppression
rows, cols = image.shape
edge_image = np.zeros((rows, cols), dtype=np.uint8)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        direction = gradient_direction[i, j]

        # Horizontal gradient
        if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
            if (gradient_magnitude[i, j] > gradient_magnitude[i, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i, j + 1]):
                edge_image[i, j] = gradient_magnitude[i, j]
        # Diagonal gradient (45 degrees)
        elif (22.5 <= direction < 67.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j + 1]):
                edge_image[i, j] = gradient_magnitude[i, j]
        # Vertical gradient
        elif (67.5 <= direction < 112.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j]):
                edge_image[i, j] = gradient_magnitude[i, j]
        # Diagonal gradient (135 degrees)
        elif (112.5 <= direction < 157.5):
            if (gradient_magnitude[i, j] > gradient_magnitude[i - 1, j + 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j - 1]):
                edge_image[i, j] = gradient_magnitude[i, j]

# Double thresholding
high_threshold = 50
low_threshold = 20

strong_edges = (edge_image > high_threshold)
weak_edges = (edge_image >= low_threshold) & (edge_image <= high_threshold)

# Edge tracking by hysteresis
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        if weak_edges[i, j]:
            if strong_edges[i - 1:i + 2, j - 1:j + 2].any():
                strong_edges[i, j] = True
                weak_edges[i, j] = False
            else:
                strong_edges[i, j] = False

# Display the edge-detected image
edge_image = (strong_edges * 255).astype(np.uint8)
cv2.imshow('Canny Edge Detection', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
