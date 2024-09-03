import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

class Calibration:
    def __init__(self, arm_ip, cam_serial, checkerboard_size=(7, 10)):
        self.arm = XArmAPI(arm_ip)
        self.arm.connect(arm_ip)
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(cam_serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        self.camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]])
        self.dist_coeffs = np.array(self.intr.coeffs)
        self.checkerboard_size = checkerboard_size
        self.square_size = 0.015
        
        # Initialize storage for robot and camera poses
        self.robot_poses = []
        self.camera_poses = []

    def move_arm(self):
        print("Move the arm to the desired position using the xArm control interface.")
        print("Press 'c' to capture the image and 3D coordinates, or 'q' to quit.")

        while len(self.robot_poses) < 30:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            # Show the camera feed with checkerboard corners overlay
            ret, corners = self.find_checkerboard_corners(color_image)
            if ret:
                cv2.drawChessboardCorners(color_image, self.checkerboard_size, corners, ret)

            cv2.imshow('Camera Feed', color_image)
            key = cv2.waitKey(1)
            if key == ord('c'):
                print("Capturing image and coordinates...")
                self.capture_image_and_pose(corners)
            elif key == ord('q'):
                print("Exiting...")
                break

        print("Collected 30 samples. Performing calibration...")
        self.calibrate_hand_eye()

    def capture_image_and_pose(self, corners):
        # Capture frames
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame or corners is None:
            print("Failed to capture frames or no corners detected.")
            return

        depth_image = np.asanyarray(depth_frame.get_data())

        # Compute 3D coordinates of corners
        camera_pose = self.get_camera_pose(corners, depth_image)

        # Get the current robot pose (end-effector pose)
        # Assuming the xArm API provides a function to get the end-effector pose directly
        self.arm.get_position(is_radian=False)
        position = self.arm.position[0:3]  # Get position (x, y, z)
        orientation = self.arm.position[3:6]  # Get orientation (roll, pitch, yaw)

        robot_pose = self.pose_to_transformation_matrix(position, orientation)

        # Store the poses
        self.robot_poses.append(robot_pose)
        self.camera_poses.append(camera_pose)
        print(f"Captured {len(self.robot_poses)} / 30")

    def find_checkerboard_corners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return ret, corners

    # def get_camera_pose(self, corners, depth_image):
    #     # Convert checkerboard corners to 3D points
    #     points = []
    #     for corner in corners:
    #         u, v = corner.ravel()
    #         z = depth_image[int(v), int(u)] / 1000.0  # Convert from mm to meters
    #         if z > 0:
    #             x = (u - 320) * z / 600.0  # Assuming intrinsic fx = 600.0 and cx = 320
    #             y = (v - 240) * z / 600.0  # Assuming intrinsic fy = 600.0 and cy = 240
    #             points.append([x, y, z])

    #     points = np.array(points)
    #     return self.estimate_pose(points)

    def get_camera_pose(self, corners, depth_image):
        # Prepare object points (3D points in the world coordinate)
        objp = np.zeros((np.prod(self.checkerboard_size), 3), np.float32)
        objp[:, :2] = np.indices(self.checkerboard_size).T.reshape(-1, 2)
        objp *= self.square_size  # Scale by the size of the checkerboard squares

        # Estimate the pose of the checkerboard
        _, rvec, tvec = cv2.solvePnP(objp, corners, self.camera_matrix, self.dist_coeffs)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.T
        return T

    def estimate_pose(self, points):
        # Placeholder for pose estimation logic (depends on the setup)
        # Typically involves solving PnP or using an existing pose estimate
        return np.eye(4)  # Returning identity as a placeholder

    def pose_to_transformation_matrix(self, position, orientation):
        # Converts position and orientation (in roll, pitch, yaw) to a 4x4 transformation matrix
        x, y, z = position
        roll, pitch, yaw = np.radians(orientation)
        print("Position:", [x, y, z])
        print("Orientation:", [roll, pitch, yaw])
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

    def calibrate_hand_eye(self):
        R_gripper2base = [pose[:3, :3] for pose in self.robot_poses]
        t_gripper2base = [pose[:3, 3] for pose in self.robot_poses]
        R_target2cam = [pose[:3, :3] for pose in self.camera_poses]
        t_target2cam = [pose[:3, 3] for pose in self.camera_poses]

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.squeeze()

        print("Calibration complete. Transformation matrix (cam2gripper):")
        print(T_cam2gripper)

    def stop(self):
        self.pipeline.stop()
        self.arm.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace '192.168.1.100' with your xArm's IP address and 'your_camera_serial' with the D405 serial number
    arm_ip = '192.168.1.197'
    cam_serial = '128422271985'
    calibrator = Calibration(arm_ip, cam_serial)
    try:
        calibrator.move_arm()
    finally:
        calibrator.stop()


