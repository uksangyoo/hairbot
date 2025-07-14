import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

class HumanDetector:
    def __init__(self, camera_serial):
        # Initialize RealSense pipeline and MediaPipe components
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

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

    def draw_axes(self, image, rotation_vector, translation_vector, length=100):
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

    def run(self):
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
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
                        # Draw the head pose in 3D directly on the original image
                        self.draw_axes(color_image, rotation_vector, translation_vector)

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
                                self.trajectory.append(avg_position)

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
                        print("Trajectory:", self.trajectory)

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
