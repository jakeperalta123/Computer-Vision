import cv2
import numpy as np

def detect_and_mark_kettle(image_paths):
    # 加載第一張照片作為水壺模板
    kettle_template = cv2.imread('target.png', 0) 

    # 創建SIFT檢測器
    sift = cv2.SIFT_create()

    # 在水壺模板上檢測關鍵點和描述子
    kp1, des1 = sift.detectAndCompute(kettle_template, None)

    # 創建BFMatcher對象
    bf = cv2.BFMatcher()

    for img_path in image_paths:
        # 讀取圖像
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 在圖像上檢測關鍵點和描述子
        kp2, des2 = sift.detectAndCompute(img_gray, None)

        # 進行特徵匹配
        matches = bf.knnMatch(des1, des2, k=2)

        # 應用比值測試來確定良好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 如果良好的匹配數量超過閾值，則認為這張照片含有水壺
        if len(good_matches) > 10:  # 可根據實際情況調整閾值
            # 提取匹配的關鍵點的坐標
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 計算透視變換矩陣
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # 提取水壺模板的尺寸
            h, w = kettle_template.shape

            # 定義水壺的四個角點
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # 透視變換水壺模板的四個角點到匹配圖像中的位置
            dst = cv2.perspectiveTransform(pts, M)

            # 在圖像上畫出水壺的框
            img = cv2.polylines(img, [np.int32(dst)], True, (0, 255, 0), 3)

            # 顯示含有水壺的照片
            cv2.imshow('Detected Kettle', img)
            cv2.waitKey(0)

# 調用函數並傳入照片的路徑列表
image_paths = []  # 添加你的照片路徑
for i in range(1, 11):
    image_paths.append(f"{i}.png")

detect_and_mark_kettle(image_paths)
