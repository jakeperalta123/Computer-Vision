import cv2
import numpy as np

# Step 1: Capture Images
left_img = cv2.imread('bottle_left.jpg')
right_img = cv2.imread('bottle_right.jpg')

# Step 2: Detect Feature Points
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)

# Step 3: Match Feature Points
bf = cv2.BFMatcher()
matches = bf.match(descriptors_left, descriptors_right)
matches = sorted(matches, key=lambda x: x.distance)

# Step 4: Compute Fundamental Matrix
points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
fundamental_matrix, _ = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC)

# Step 5: Rectify Images
# Step 5: Rectify Images
_, H1, H2 = cv2.stereoRectifyUncalibrated(points_left, points_right, fundamental_matrix, left_gray.shape[:2])
rectified_left = cv2.warpPerspective(left_gray, H1, left_gray.shape[::-1])
rectified_right = cv2.warpPerspective(right_gray, H2, right_gray.shape[::-1])

# Step 6: Compute Disparity
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(rectified_left, rectified_right)

# Convert disparity to absolute values to avoid negative values
disparity = np.abs(disparity)

# Step 7: Compute Depth
baseline = 0.61  # Example baseline distance (in meters)
focal_length = 1600  # Example focal length (in pixels)
depth_map = baseline * focal_length / (disparity + 0.0001)

# Step 8: Measure Distance
# Choose a point corresponding to the marker in the depth map
marker_depth = depth_map[2594, 2323]  # Coordinates of the marker in the depth map
# Measure distance D (distance from camera to marker)
D = marker_depth
print("Estimated distance D:", D)

