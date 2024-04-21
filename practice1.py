import depthai as dai
import cv2

# 初始化 DepthAI 设备
device = dai.Device()

# 创建左相机流
left_camera = device.createMonoCamera()
left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)

# 创建右相机流
right_camera = device.createMonoCamera()
right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# 启动设备
device.startPipeline()

# 获取左/右相机输出流队列
left_queue = device.getOutputQueue(name="left", maxSize=1, blocking=False)
right_queue = device.getOutputQueue(name="right", maxSize=1, blocking=False)

while True:
    # 获取左相机图像
    left_frame = left_queue.get().getCvFrame()

    # 获取右相机图像
    right_frame = right_queue.get().getCvFrame()

    # 在窗口中显示左/右相机图像
    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Right Camera", right_frame)

    # 检查是否按下 ESC 键，如果是则退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放资源并关闭窗口
cv2.destroyAllWindows()
device.close()
