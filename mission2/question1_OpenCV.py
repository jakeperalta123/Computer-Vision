import cv2

# Read the image
image = cv2.imread('mission2A.png', cv2.IMREAD_GRAYSCALE)

# Perform Canny edge detection
edges = cv2.Canny(image, 100, 200)  # You can adjust the thresholds as needed

# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
