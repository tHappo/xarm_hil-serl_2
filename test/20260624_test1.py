from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.100"

arm = XArmAPI(ROBOT_IP)
print("connected:", arm.connected)
print("state:", arm.state)
print("error_code:", arm.error_code)
print("warn_code:", arm.warn_code)

code, pose = arm.get_position()
print("get_position code:", code)
print("pose:", pose)

arm.disconnect()
