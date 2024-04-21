import cv2
import numpy as np

# Load the images
image_paths = []
for i in range(1, 11):
    image_paths.append(f"{i}.png")


images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors for each image
keypoints_list = []
descriptors_list = []

for img in images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints_list.append(kp)
    descriptors_list.append(des)

# Initialize BFMatcher
bf = cv2.BFMatcher()

# Match descriptors between consecutive images
matches_list = []

for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors_list[i], descriptors_list[i+1], k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    matches_list.append(good_matches)

# Draw matches
for i, matches in enumerate(matches_list):
    img1 = images[i]
    img2 = images[i+1]
    kp1 = keypoints_list[i]
    kp2 = keypoints_list[i+1]
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow(f'Matches between Image {i+1} and Image {i+2}', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
