from easygopigo3 import EasyGoPiGo3
import time

robot = EasyGoPiGo3()
parking_spot_status = {}
distance_sensor = robot.init_distance_sensor()  # 初始化距離感測器

for i in range (10):
    spot_name = f"Spot {i}"
    robot.drive_cm(500)
    robot.turn_degree(90)
    distance = distance_sensor.read_mm()
    if (distance <= 1000):
        parking_spot_status[spot_name] = "Occupied"
    else:
        parking_spot_status[spot_name] = "Empty"
    robot.turn_degree(-90)

print(parking_spot_status)