import cv2 
import matplotlib.pyplot as plt
import numpy as np


# read images
img1 = cv2.imread('tower1.png')  
img2 = cv2.imread('tower2.png') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()

ssd = 0
for match in matches:
    descriptor_1 = descriptors_1[match.queryIdx]
    descriptor_2 = descriptors_2[match.trainIdx]
    ssd += np.sum((descriptor_1 - descriptor_2) ** 2)

print("SSD value:", ssd)

# Compute Homography matrix
src_pts = np.float32([keypoints_1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Compute its inverse
inverse_homography = np.linalg.inv(homography)

print("Homography matrix:\n", homography)
print("Inverse Homography matrix:\n", inverse_homography)
