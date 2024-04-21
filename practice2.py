from depthai_sdk import OakCamera
from depthai import ColorCameraProperties, MonoCameraProperties

with OakCamera() as oak:
    color = oak.create_camera('color', resolution=ColorCameraProperties.SensorResolution.THE_1080_P, fps=35)
    stereo = oak.create_stereo(resolution=MonoCameraProperties.SensorResolution.THE_480_P, fps=99)
    oak.visualize([color, stereo])
    oak.start(blocking=True)

    while oak.running():
        oak.poll()
