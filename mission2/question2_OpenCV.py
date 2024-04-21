import cv2

# Read the image
image = cv2.imread('mission2B.png', cv2.IMREAD_GRAYSCALE)

# Perform Harris corner detection
dst = cv2.cornerHarris(image, 2, 3, 0.04)  # You can adjust parameters as needed

# Threshold the corner response
threshold = 0.01 * dst.max()
corners = cv2.threshold(dst, threshold, 255, cv2.THRESH_BINARY)[1]

# Display the original image and detected corners
cv2.imshow('Original Image', image)
cv2.imshow('Detected Corners', corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
