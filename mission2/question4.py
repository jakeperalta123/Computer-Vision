import cv2
import numpy as np

# Function to compute integral image manually
def compute_integral_image(img):
    integral_img = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            integral_img[y, x] = img[:y+1, :x+1].sum()
    return integral_img

# Load image
image_path = 'tower.png'  # Change to your image file path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Unable to read image.")
    exit()

# Convert frame to grayscale for integral image computation
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Compute integral image
integral_img = compute_integral_image(gray_frame)

# Convert integral image to uint8 for display
integral_img_display = cv2.normalize(integral_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display both original RGB image and integral image
cv2.imshow('RGB Image', frame)
cv2.imshow('Integral Image', integral_img_display)

# Wait for any key to be pressed
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()
