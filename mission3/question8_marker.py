import cv2
import apriltag

def main():
    # 初始化AprilTag檢測器
    detector = apriltag.Detector()

    # 初始化視頻捕獲
    cap = cv2.VideoCapture(0)

    # 主循環
    while True:
        # 讀取一幀圖像
        ret, frame = cap.read()

        # 將圖像轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度圖上檢測AprilTag
        detections, _ = detector.detect(gray)

        # 遍歷檢測到的AprilTag
        for detection in detections:
            # 獲取AprilTag的角點
            corners = detection.corners.astype(int)

            # 在圖像上繪製AprilTag的外框
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

            # 獲取AprilTag的ID
            tag_id = detection.tag_id
            # 在圖像上顯示AprilTag的ID
            cv2.putText(frame, str(tag_id), (corners[0, 0], corners[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 顯示圖像
        cv2.imshow('AprilTag Detection', frame)

        # 檢查退出條件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
