from utils.util_ft import FTSensor
from utils.util_realsense import Camera
from xarm.wrapper import XArmAPI
import numpy as np
import open3d as o3d



def pose_to_transformation_matrix(position, orientation):
    # Converts position and orientation (in roll, pitch, yaw) to a 4x4 transformation matrix
    x, y, z = position
    roll, pitch, yaw = np.radians(orientation)
    # print("Position:", [x, y, z])
    # print("Orientation:", [roll, pitch, yaw])
    # Compute rotation matrix from roll, pitch, yaw
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    # Combine rotation matrix and position into a transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x/1000, y/1000, z/1000]

    return T





ftsensor = FTSensor()
hand_cam = Camera('128422271985')

arm_ip = '192.168.1.197'
arm = XArmAPI(arm_ip)
arm.connect(arm_ip)
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

data_root = '/home/frida/hairbot/data/'

try:
    for i in range(30):  # Example: Get 10 readings

        #press c to continue
        input("Press Enter to continue...")

        position = arm.position[0:3]  # Get position (x, y, z)
        orientation = arm.position[3:6]  # Get orientation (roll, pitch, yaw)
        wrenches = []
        for _ in range(100):
            force_torque = ftsensor.get_ft()
            wrenches.append(force_torque)
        wrenches = np.array(wrenches)
        # print("wrench shape ", wrenches.shape)
        pcd = hand_cam.get_pcd()
        robot_pose = pose_to_transformation_matrix(position, orientation)

        # Save data
        np.save(data_root + 'wrench/' + str(i) + '.npy', wrenches)
        o3d.io.write_point_cloud(data_root + 'pcd/' + str(i) + '.pcd', pcd)
        np.save(data_root + 'pose/' + str(i) + '.npy', robot_pose)





except:
    ftsensor.stop()
    hand_cam.stop()
    arm.disconnect()

ftsensor.stop()
hand_cam.stop()
arm.disconnect()

