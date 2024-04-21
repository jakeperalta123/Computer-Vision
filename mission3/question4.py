import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video
video_path = 'video.MOV'
cap = cv2.VideoCapture(video_path)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Color for drawing optical flow tracks
color = (0, 255, 0)

def plot_optical_flow(frame1, frame2, flow, step, title):
    h, w, _ = frame1.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(frame1[:, :, ::-1])
    for (x1, y1), (x2, y2) in lines:
        plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=2)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Frame counter
frame_count = 0

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Plot optical flow
    if frame_count % 1 == 0:  # Modify this value based on (i), (ii), (iii)
        plot_optical_flow(prev_frame, frame, flow, step=15, title=f'Frame {frame_count}')
    
    # Update previous frame
    prev_gray = gray
    prev_frame = frame

# Release video capture
cap.release()
cv2.destroyAllWindows()
