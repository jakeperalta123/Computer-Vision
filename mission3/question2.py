import cv2

def calculate_optical_flow(frame1, frame2):
    # Convert frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute gradients
    gradient_x1 = cv2.Sobel(frame1_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y1 = cv2.Sobel(frame1_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x2 = cv2.Sobel(frame2_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y2 = cv2.Sobel(frame2_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Select feature points
    corners = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, corners, None, **lk_params)

    # Calculate motion function estimates
    u = (p1[..., 0] - corners[..., 0]).reshape(-1, 1)
    v = (p1[..., 1] - corners[..., 1]).reshape(-1, 1)

    return u, v

frame1 = cv2.imread('frame1.jpg')
frame2 = cv2.imread('frame2.jpg')
u, v = calculate_optical_flow(frame1, frame2)
print(u, v)
