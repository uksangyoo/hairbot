import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from util_robot import xArmRobot
def relative_transformation(initial_transform, final_transform):
    """
    Faster calculation of the relative transformation to get from the initial to the final transformation.
    
    Parameters:
    - initial_transform: np.array of shape (4, 4) representing the initial 4x4 transformation matrix.
    - final_transform: np.array of shape (4, 4) representing the final 4x4 transformation matrix.
    
    Returns:
    - relative_transform: np.array of shape (4, 4) representing the transformation from the initial to the final pose.
    """
    # Extract the rotation (3x3) and translation (3x1) components from the initial transform
    initial_rotation = initial_transform[:3, :3]
    initial_translation = initial_transform[:3, 3]
    
    # Compute the inverse rotation (transpose of the rotation matrix)
    initial_rotation_inv = initial_rotation.T
    
    # Compute the inverse translation
    initial_translation_inv = -np.dot(initial_rotation_inv, initial_translation)
    
    # Extract rotation and translation from the final transform
    final_rotation = final_transform[:3, :3]
    final_translation = final_transform[:3, 3]
    
    # Compute the relative rotation
    relative_rotation = np.dot(initial_rotation_inv, final_rotation)
    
    # Compute the relative translation
    relative_translation = np.dot(initial_rotation_inv, final_translation) + initial_translation_inv
    
    # Construct the relative transformation matrix
    relative_transform = np.eye(4)
    relative_transform[:3, :3] = relative_rotation
    relative_transform[:3, 3] = relative_translation#/1000 # Convert translation to meters
    
    return relative_transform

def rotation_vector_to_transformation_matrix(rotation_vector, translation_vector):
    """
    Converts a rotation vector and translation vector into a 4x4 transformation matrix.
    
    Parameters:
    - rotation_vector: np.array of shape (3,) representing the rotation as an axis-angle vector.
    - translation_vector: np.array of shape (3,) representing the translation in x, y, z coordinates.
    
    Returns:
    - transformation_matrix: np.array of shape (4, 4) representing the resulting 4x4 transformation matrix.
    """
    # Convert rotation vector (axis-angle) to a 3x3 rotation matrix using Rodrigues' formula
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Initialize the transformation matrix as a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    
    # Set the top-left 3x3 part of the transformation matrix to the rotation matrix
    transformation_matrix[:3, :3] = rotation_matrix
    
    # Set the top-right 3x1 part of the transformation matrix to the translation vector
    transformation_matrix[:3, 3] = np.squeeze(translation_vector)
    
    return transformation_matrix

def transform_position(position, transformation_matrix):
    """
    Transforms a 3D position (x, y, z) by a 4x4 transformation matrix.
    
    Parameters:
    - position: np.array of shape (3,) representing the (x, y, z) position.
    - transformation_matrix: np.array of shape (4, 4) representing the transformation matrix.
    
    Returns:
    - transformed_position: np.array of shape (3,) representing the transformed (x', y', z') position.
    """
    # Convert the position (x, y, z) to homogeneous coordinates (x, y, z, 1)
    homogeneous_position = np.append(position, 1)
    
    # Apply the transformation matrix
    transformed_homogeneous = np.dot(transformation_matrix, homogeneous_position)
    
    # Convert back from homogeneous coordinates (ignore the 4th element)
    transformed_position = transformed_homogeneous[:3]
    
    return transformed_position
def transform_positions(positions, transformation_matrix):
    """
    Transforms a list of 3D positions (x, y, z) by a 4x4 transformation matrix and returns the result as a list.
    
    Parameters:
    - positions: list or np.array of shape (N, 3) where each row represents a 3D position (x, y, z).
    - transformation_matrix: np.array of shape (4, 4) representing the transformation matrix.
    
    Returns:
    - transformed_positions_list: list of lists where each inner list is the transformed 3D position [x', y', z'].
    """
    # Convert positions to homogeneous coordinates (N, 4) by adding a column of ones
    positions= np.array(positions)
    homogeneous_positions = np.hstack([positions, np.ones((positions.shape[0], 1))])
    
    # Apply the transformation matrix to each position
    transformed_homogeneous = np.dot(transformation_matrix, homogeneous_positions.T).T
    
    # Convert back from homogeneous coordinates (ignore the 4th element)
    transformed_positions = transformed_homogeneous[:, :3]
    
    # Convert the result to a Python list of lists
    transformed_positions_list = transformed_positions.tolist()
    
    return transformed_positions_list

class HumanDetector:
    def __init__(self, camera_serial):
        # Initialize RealSense pipeline and MediaPipe components
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.head_pose = None
        # Get camera intrinsics
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ], dtype="float64")
        self.dist_coeffs = np.array(self.intrinsics.coeffs[:4])

        # Initialize MediaPipe FaceMesh and Hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)


        self.robot_trajectory = None
        arm_ip = '192.168.1.197'
        self.robot = xArmRobot(arm_ip)
        self.color_frame = None

        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ])

        self.track_left_hand = False
        self.record_trajectory = False  # Flag to toggle recording
        self.trajectory = []  # List to store the trajectory of average 3D positions

    def get_3d_point(self, depth_frame, x, y):
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        if 0 <= x < depth_intrinsics.width and 0 <= y < depth_intrinsics.height:
            depth_value = depth_frame.get_distance(x, y)
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth_value)
                return np.array(point_3d)
        return None

    def calculate_average_position(self, points):
        """
        Calculate the average 3D position of a list of points.
        """
        valid_points = [p for p in points if p is not None]
        if valid_points:
            return np.mean(valid_points, axis=0)
        return None

    def project_3d_to_2d(self, point_3d):
        """
        Project a 3D point to 2D image coordinates using the camera matrix and distortion coefficients.
        """
        point_2d, _ = cv2.projectPoints(np.array([point_3d]), np.zeros((3, 1)), np.zeros((3, 1)), self.camera_matrix, self.dist_coeffs)
        return tuple(map(int, point_2d.ravel()))

    def draw_axes(self, image, rotation_vector, translation_vector, length=500):
        axis_points = np.float32([
            [length, 0, 0],   # X-axis (Red)
            [0, length, 0],   # Y-axis (Green)
            [0, 0, length]    # Z-axis (Blue)
        ]).reshape(-1, 3)

        axis_points_2d, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)
        origin_2d, _ = cv2.projectPoints(np.zeros((3, 1)), rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs)

        origin = tuple(map(int, origin_2d[0].ravel()))
        x_axis = tuple(map(int, axis_points_2d[0].ravel()))
        y_axis = tuple(map(int, axis_points_2d[1].ravel()))
        z_axis = tuple(map(int, axis_points_2d[2].ravel()))

        cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X-axis (Red)
        cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis (Green)
        cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z-axis (Blue)

    def draw_hand_position(self, image, hand_landmarks):
        for landmark in hand_landmarks:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw green circles on hand landmarks

    def draw_trajectory(self, image):
        """
        Draw the recorded 3D trajectory points on the 2D image.
        """
        for point_3d in self.trajectory:
            point_2d = self.project_3d_to_2d(point_3d)
            cv2.circle(image, point_2d, 3, (255, 0, 0), -1)  # Draw trajectory points in blue

    
    def find_aruco_marker_transformation(self, marker_id=2, marker_size=0.07):
        """Detect the ArUco marker and return its transformation matrix (rotation and translation)."""
        #frames = self.pipeline.wait_for_frames()
        color_frame = self.color_frame

        color_image = np.asanyarray(color_frame.get_data())

        # Define the ArUco dictionary and parameters
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters_create()

        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        # If the marker with the given ID is detected
        if ids is not None and marker_id in ids:
            # Get the index of the marker
            marker_index = np.where(ids == marker_id)[0][0]

            # Estimate pose of the marker
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                      [0, intrinsics.fy, intrinsics.ppy],
                                      [0, 0, 1]])
            dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker_index], marker_size, camera_matrix, dist_coeffs)

            # Extract rotation and translation vectors
            rvec = rvecs[0]
            tvec = tvecs[0]

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Create the transformation matrix (4x4)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.flatten()

            return transformation_matrix
        else:
            print(f"Marker with ID {marker_id} not found.")
            return None




    def run(self):
        try:
            
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                self.color_frame = frames.get_color_frame()

                if not depth_frame or not self.color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(self.color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Process the image with MediaPipe FaceMesh
                face_results = self.face_mesh.process(rgb_image)

                # Track the head pose using FaceMesh
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]

                    # Extract 2D image points corresponding to 3D model points
                    image_points = np.array([
                        (face_landmarks.landmark[1].x * color_image.shape[1], face_landmarks.landmark[1].y * color_image.shape[0]),  # Nose tip
                        (face_landmarks.landmark[152].x * color_image.shape[1], face_landmarks.landmark[152].y * color_image.shape[0]),  # Chin
                        (face_landmarks.landmark[33].x * color_image.shape[1], face_landmarks.landmark[33].y * color_image.shape[0]),  # Left eye
                        (face_landmarks.landmark[263].x * color_image.shape[1], face_landmarks.landmark[263].y * color_image.shape[0]),  # Right eye
                        (face_landmarks.landmark[61].x * color_image.shape[1], face_landmarks.landmark[61].y * color_image.shape[0]),  # Left mouth corner
                        (face_landmarks.landmark[291].x * color_image.shape[1], face_landmarks.landmark[291].y * color_image.shape[0])  # Right mouth corner
                    ], dtype="double")

                    # SolvePnP to get head pose
                    success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

                    if success:
                        # Ensure translation_vector is used in meters without any conversion to mm
                        # Make sure no scaling happens here
                        self.draw_axes(color_image, rotation_vector, translation_vector)

                        # If you store or use the translation_vector elsewhere, it should stay in meters
                        if self.head_pose is None:
                            self.head_pose = rotation_vector_to_transformation_matrix(rotation_vector, translation_vector)
                        head_pose_current = rotation_vector_to_transformation_matrix(rotation_vector, translation_vector)
                # Process the image with MediaPipe Hands
                hand_results = self.hands.process(rgb_image)

                # Track the hand
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        if hand_handedness.classification[0].label == "Left":
                            # Draw 2D landmarks on the hand
                            self.draw_hand_position(color_image, hand_landmarks.landmark)

                            # Get the 3D positions of the three middle knuckles (middle, ring, pinky)
                            middle_knuckle = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                            ring_knuckle = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                            pinky_knuckle = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

                            middle_3d = self.get_3d_point(depth_frame, int(middle_knuckle.x * color_image.shape[1]), int(middle_knuckle.y * color_image.shape[0]))
                            ring_3d = self.get_3d_point(depth_frame, int(ring_knuckle.x * color_image.shape[1]), int(ring_knuckle.y * color_image.shape[0]))
                            pinky_3d = self.get_3d_point(depth_frame, int(pinky_knuckle.x * color_image.shape[1]), int(pinky_knuckle.y * color_image.shape[0]))

                            # Calculate the average 3D position
                            avg_position = self.calculate_average_position([middle_3d, ring_3d, pinky_3d])

                            # If recording is enabled, store the average position in the trajectory
                            if self.record_trajectory and avg_position is not None:
                                head_relative_transform = relative_transformation(self.head_pose, head_pose_current)
                                self.trajectory.append(avg_position)
                                trajectory_T0 = self.trajectory
                                # self.trajectory = transform_positions(self.trajectory, head_relative_transform)
                                self.head_pose = head_pose_current
                                head_pose_recording = head_pose_current

                if not self.record_trajectory and len(self.trajectory) > 0:
                    head_relative_transform = relative_transformation(head_pose_current,head_pose_recording)
                    head_relative_transform = np.eye(4)
                    #print("Head Pose Movement", head_relative_transform)
                    self.trajectory = transform_positions(trajectory_T0, head_relative_transform)
                    self.head_pose = head_pose_current

                # Draw the recorded trajectory on the image
                self.draw_trajectory(color_image)

                # Display the camera feed with overlays (head pose, hand tracking, and trajectory)
                cv2.imshow('Real-Time Hand and Head Tracking with Trajectory', color_image)

                # Keyboard input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit when 'q' is pressed
                    break
                elif key == ord('r'):  # Toggle recording when 'r' is pressed
                    self.record_trajectory = not self.record_trajectory
                    if self.record_trajectory:
                        print("Recording started...")
                    else:
                        print("Recording stopped...")
                        T_third2marker = self.find_aruco_marker_transformation()
                        print("T_third2marker:", T_third2marker)
                        self.robot_trajectory = transform_positions(self.trajectory, np.linalg.inv(T_third2marker))
                        T_base2marker = np.load('/home/frida/Projects/hairbot/scripts/camera_calibration/base2marker.npy')
                        T_marker2base = np.linalg.inv(T_base2marker)
                        self.robot_trajectory = transform_positions(self.robot_trajectory, T_marker2base)
                        print("Robot Trajectory:", self.robot_trajectory)
                elif key == ord('t'):
                    self.robot.follow_xyz_trajectory(self.robot_trajectory, speed=300, wait=True)

        finally:
            # Stop streaming
            self.pipeline.stop()
            self.face_mesh.close()
            self.hands.close()
            cv2.destroyAllWindows()

# To run the HumanDetector
if __name__ == "__main__":
    camera_serial = '821212060490'  # Replace with your RealSense camera serial number
    detector = HumanDetector(camera_serial)
    detector.run()
