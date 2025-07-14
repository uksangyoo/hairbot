from xarm.wrapper import XArmAPI
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from sklearn.linear_model import RANSACRegressor
def fit_plane_and_project(points):
    """
    Fit a plane to a set of 3D points, reject outliers, and project the inlier points onto the plane.

    Args:
        points (numpy.ndarray): Nx3 array of 3D points.

    Returns:
        projected_points (numpy.ndarray): Nx3 array of 3D points projected onto the fitted plane.
    """
    # Ensure points is an Nx3 array
    points = np.asarray(points)
    assert points.shape[1] == 3, "Input should be an Nx3 array of 3D points"

    # Define the plane as z = ax + by + c
    X = points[:, :2]  # Take only x and y as independent variables
    Z = points[:, 2]   # z is the dependent variable

    # Use RANSAC to fit the plane and reject outliers
    ransac = RANSACRegressor()
    ransac.fit(X, Z)
    
    # Get the inliers and coefficients of the plane
    inlier_mask = ransac.inlier_mask_
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # Plane normal vector is [a, b, -1] based on the equation z = ax + by + c
    normal_vector = np.array([a, b, -1])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector

    # Select the inlier points
    inlier_points = points[inlier_mask]

    # Point on the plane (choosing the first inlier point for simplicity)
    point_on_plane = np.array([inlier_points[0][0], inlier_points[0][1], a*inlier_points[0][0] + b*inlier_points[0][1] + c])

    # Project the points onto the plane
    projected_points = np.empty_like(inlier_points)

    for i, point in enumerate(inlier_points):
        # Vector from the point to a point on the plane
        vec_to_plane = point - point_on_plane
        
        # Project the vector onto the normal vector
        distance_to_plane = np.dot(vec_to_plane, normal_vector)
        
        # Subtract the normal component to project onto the plane
        projected_points[i] = point - distance_to_plane * normal_vector

    return projected_points

# Example usage:
class xArmRobot:
    def __init__(self, ip_address):
        # Initialize the robot connection using the given IP address
        self.arm = XArmAPI(ip_address)
        self.arm.connect()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

        # Check connection status
        if self.arm.error_code != 0:
            print(f"Error connecting to xArm with IP: {ip_address}, Error Code: {self.arm.error_code}")
        else:
            print(f"Successfully connected to xArm with IP: {ip_address}")

        #set position to home
        self.move_to_position([53.2, -169, 425, 180, -10, -180], speed=100, wait=True)
    
    def get_position(self):
        """Returns the current position of the xArm end-effector as a list [x, y, z, roll, pitch, yaw]"""
        position = self.arm.position
        return position
    def move_to_home(self):
        """
        Moves the xArm to the home position.
        """
        self.move_to_position([53.2, -169, 425, 180, -10, -180], speed=100, wait=True)
        
    def move_to_position(self, position, speed=100, wait=True):
        """
        Moves the xArm to the specified position.
        :param position: A list [x, y, z, roll, pitch, yaw] representing the target position
        :param speed: The speed to move the arm at
        :param wait: Whether to wait for the movement to finish before continuing
        """
        if len(position) != 6:
            raise ValueError("Position must be a list of 6 elements [x, y, z, roll, pitch, yaw]")
        self.arm.set_position(*position, speed=speed, wait=wait)

    def follow_trajectory(self, trajectory, speed=100, wait=True):
        """
        Moves the xArm along a trajectory of positions.
        :param trajectory: A list of positions where each position is a list [x, y, z, roll, pitch, yaw]
        :param speed: The speed to move the arm at
        :param wait: Whether to wait for the movement to finish before continuing
        """
        for position in trajectory:
            self.move_to_position(position, speed=speed, wait=wait)
            if wait:
                time.sleep(0.5)  # Optional delay between trajectory points

    def move_by(self, dx, dy, dz, speed=100, wait=True):
        """
        Moves the xArm by a relative offset in the XYZ directions.
        :param dx: The offset in the X direction
        :param dy: The offset in the Y direction
        :param dz: The offset in the Z direction
        :param speed: The speed to move the arm at
        :param wait: Whether to wait for the movement to finish before continuing
        """
        current_position = self.get_position()
        new_position = [
            current_position[0] + dx,
            current_position[1] + dy,
            current_position[2] + dz,
            current_position[3],  # Roll remains the same
            current_position[4],  # Pitch remains the same
            current_position[5]   # Yaw remains the same
        ]
        self.move_to_position(new_position, speed=speed, wait=wait)
    def follow_xyz_trajectory(self, xyz_trajectory, speed=10, wait=True):
        """
        Moves the xArm along a trajectory of XYZ positions with auto-calculated roll, pitch, and yaw to keep
        the end-effector normal to the trajectory.
        :param xyz_trajectory: A Nx3 array of positions where each position is [x, y, z]
        :param speed: The speed to move the arm at
        :param wait: Whether to wait for the movement to finish before continuing
        """
        if len(xyz_trajectory) < 2:
            raise ValueError("Trajectory must contain at least two positions.")
        

        #fit to a plane
        xyz_trajectory = np.array(xyz_trajectory)
        xyz_trajectory = xyz_trajectory[:, :3]
        projected_points = fit_plane_and_project(xyz_trajectory)

        for i in range(projected_points.shape[0]):
            # Current and next point in the trajectory
            current_point = np.array(projected_points[i,:])*1000
            print("current point: " ,current_point)
            next_point = np.array(xyz_trajectory[i + 1])
            
            # Calculate the direction vector (next_point - current_point)
            direction_vector = next_point - current_point
            direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector

            # Calculate roll, pitch, yaw to make the end-effector normal to the direction
            # Assuming the roll is 180 degrees (upward), we can calculate pitch and yaw
            # from the direction vector. This assumes a simple transformation.
            yaw = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi  # Yaw: rotation around Z-axis
            pitch = np.arctan2(-direction_vector[2], np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)) * 180 / np.pi  # Pitch: rotation around Y-axis
            roll = 180  # Keep the end-effector always facing upwards

            # Combine XYZ with the calculated roll, pitch, and yaw
            target_position = [current_point[0], current_point[1], current_point[2]-20,  180, -10, -180]
            self.move_to_position(target_position, speed=speed, wait=wait)
            
            # if wait:
            #     time.sleep(0.5)  # Optional delay between trajectory points


    def follow_smoothed_xyz_trajectory(self, xyz_trajectory, num_points=100, speed=100, wait=True):
        """
        Moves the xArm along a smoothed and evenly spaced trajectory of XYZ positions with auto-calculated
        roll, pitch, and yaw to keep the end-effector normal to the trajectory.
        :param xyz_trajectory: A Nx3 array of positions where each position is [x, y, z]
        :param num_points: The number of points for the smoothed trajectory (default is 100)
        :param speed: The speed to move the arm at
        :param wait: Whether to wait for the movement to finish before continuing
        """
        # Ensure that xyz_trajectory is a numpy array
        xyz_trajectory = np.array(xyz_trajectory)*1000
        
        if len(xyz_trajectory) < 2:
            raise ValueError("Trajectory must contain at least two positions.")
        
        # First, calculate the cumulative distance along the trajectory
        distances = [0]
        for i in range(1, len(xyz_trajectory)):
            dist = euclidean(xyz_trajectory[i], xyz_trajectory[i-1])
            distances.append(distances[-1] + dist)
        
        # Create an interpolation function for each axis
        distances = np.array(distances)
        x_interp = interp1d(distances, xyz_trajectory[:, 0], kind='linear')
        y_interp = interp1d(distances, xyz_trajectory[:, 1], kind='linear')
        z_interp = interp1d(distances, xyz_trajectory[:, 2], kind='linear')
        
        # Generate evenly spaced points along the trajectory
        total_distance = distances[-1]
        new_distances = np.linspace(0, total_distance, num_points)
        
        # Interpolated XYZ positions
        smoothed_xyz_trajectory = np.stack([x_interp(new_distances), y_interp(new_distances), z_interp(new_distances)], axis=-1)
        
        # Stop the loop before the last index to avoid out-of-bounds errors
        for i in range(len(smoothed_xyz_trajectory) - 1):
            # Current and next point in the smoothed trajectory
            current_point = np.array(smoothed_xyz_trajectory[i])
            next_point = np.array(smoothed_xyz_trajectory[i + 1])
            
            # Calculate the direction vector (next_point - current_point)
            direction_vector = next_point - current_point
            direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector

            # Calculate roll, pitch, yaw to make the end-effector normal to the direction
            yaw = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi  # Yaw: rotation around Z-axis
            pitch = np.arctan2(-direction_vector[2], np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)) * 180 / np.pi  # Pitch: rotation around Y-axis
            roll = 180  # Keep the end-effector always facing upwards

            # Combine XYZ with the calculated roll, pitch, and yaw
            target_position = [current_point[0], current_point[1], current_point[2]-20, -179, -30, 90]
            #target_position = [current_point[0], current_point[1], current_point[2], roll, pitch, yaw]
            self.move_to_position(target_position, speed=speed, wait=wait)

        current_position = self.get_position()
        current_position[2] = current_position[2] + 120
        self.move_to_position(current_position, speed=speed, wait=wait)
        current_position = self.get_position()
        current_position[0] = current_position[0] - 120
        self.move_to_position(current_position, speed=speed, wait=wait)
