import cv2
import numpy as np

def image_stitch(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector (SIFT)
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Initialize feature matcher (FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Check if enough good matches are found
    if len(good_matches) < 10:
        return None
    
    # Extract matched keypoints
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    # Warp image1 to image2
    height, width = image2.shape[:2]
    warped_image = cv2.warpPerspective(image1, H, (width*2, height))
    
    # Blend warped_image with image2
    result = np.zeros((height, width*2, 3), dtype=np.uint8)
    result[:, :width] = image2
    result[:, width:] = warped_image[:, width:]
    
    return result

# Test the function
image1 = cv2.imread('stich1.jpg')  # Change to your image file path
image2 = cv2.imread('stich2.jpg')  # Change to your image file path

panorama = image_stitch(image1, image2)

if panorama is not None:
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches found to create a panorama.")
