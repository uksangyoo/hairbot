import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def remove_depth_outliers(depth_image, nb_neighbors=20, std_ratio=3.0):
    """
    Remove statistical outliers from the depth image by zeroing them out.
    
    Args:
    depth_image (numpy.ndarray): The input depth image (2D array).
    nb_neighbors (int): Number of neighbors to analyze for each point.
    std_ratio (float): Standard deviation ratio to define outliers.
    
    Returns:
    numpy.ndarray: Depth image with outliers removed (zeroed out).
    """
    # Reshape the depth image to a 1D array of depth values
    depth_values = depth_image.flatten()
    
    # Ignore zero values (assuming 0 represents no depth data)
    valid_depth_values = depth_values[depth_values > 0]
    
    # Compute the mean and standard deviation of the depth values
    mean_depth = np.mean(valid_depth_values)
    std_depth = np.std(valid_depth_values)
    
    # Determine the threshold for removing outliers
    threshold_lower = mean_depth - std_ratio * std_depth
    threshold_upper = mean_depth + std_ratio * std_depth
    
    # Zero out the outliers
    outlier_mask = (depth_values < threshold_lower) | (depth_values > threshold_upper)
    depth_values[outlier_mask] = 0
    
    # Reshape the 1D array back to the original depth image shape
    filtered_depth_image = depth_values.reshape(depth_image.shape)
    
    return filtered_depth_image

# Example usage:
# Assuming depth_image is a 2D numpy array representing the depth image
# filtered_depth_image = remove_depth_outliers(depth_image)

# Display the filtered depth image
# depth_visual = cv2.normalize(filtered_depth_image, None, 0, 255, cv2.NORM_MINMAX)
# depth_visual = np.uint8(depth_visual)
# cv2.imshow("Filtered Depth Image", depth_visual)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def visualize_mask_on_image(image, mask, mask_color=(0, 255, 0), alpha=0.5, fade_factor=0.3):
    """
    Visualize a binary mask over an RGB image, with the masked area highlighted and the background faded.
    
    Args:
    image (numpy.ndarray): The input RGB image (shape: HxWx3).
    mask (numpy.ndarray): The binary mask (shape: HxW) with 1 for the region to highlight and 0 for the background.
    mask_color (tuple): The color to use for highlighting the masked area (BGR format).
    alpha (float): The transparency of the mask highlight.
    fade_factor (float): The factor by which to fade the background (0 to 1, lower means more fade).
    
    Returns:
    numpy.ndarray: The resulting image with the mask visualized.
    """
    # Ensure the mask is binary (0 or 1)
    mask = mask.astype(np.uint8)

    # Create a color mask from the binary mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = mask_color

    # Highlight the masked area by blending the colored mask with the image
    highlighted_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    # Fade the background by multiplying it with the fade factor
    faded_image = image.copy()
    faded_image[mask == 0] = (faded_image[mask == 0] * fade_factor).astype(np.uint8)

    # Combine the highlighted area with the faded background
    final_image = np.where(mask[:, :, None] == 1, highlighted_image, faded_image)

    return final_image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_depth_image_blues(depth_image):
    """
    Visualize a depth image using the 'Blues' colormap from matplotlib, where closer areas are whiter
    and further areas are bluer.
    
    Args:
    depth_image (numpy.ndarray): The input depth image (2D array).
    
    Returns:
    numpy.ndarray: The RGB image with the depth values visualized using the 'Blues' colormap.
    """
    # Normalize the depth image to the range [0, 1] for colormap application

    #depth_image[depth_image > 200] = 0
    print(np.max(depth_image))  
    normalized_depth = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)
    
    # Get the 'Blues' colormap from matplotlib
    colormap = cm.get_cmap('Blues')
    
    # Apply the colormap to the normalized depth image
    colored_depth_image = colormap(normalized_depth)
    
    # Convert the colormap output (which is RGBA) to RGB
    colored_depth_image = (colored_depth_image[:, :, :3] * 255).astype(np.uint8)
    #display the image
    plt.imshow(colored_depth_image)
    plt.show()
    #return colored_depth_image

def inverse_transform(matrix):
    """
    Compute the inverse of a 4x4 transformation matrix (assumed to represent
    a rigid-body transformation, i.e., a combination of rotation and translation).
    
    Args:
    matrix (numpy.ndarray): 4x4 transformation matrix
    
    Returns:
    numpy.ndarray: 4x4 inverse of the transformation matrix
    """
    # Extract the rotation (upper-left 3x3 part)
    R = matrix[:3, :3]
    
    # Extract the translation (first 3 elements of the fourth column)
    t = matrix[:3, 3]
    
    # Compute the inverse of the rotation (transpose for an orthogonal matrix)
    R_inv = R.T
    
    # Compute the inverse translation
    t_inv = -R_inv @ t
    
    # Create the inverse transformation matrix
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = t_inv
    
    return inv_matrix


class Camera:
    def __init__(self, serial):
        """Initialize the camera with the given serial number."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.device = None
        self.sam = None
        # Start streaming
        self.pipeline.start(self.config)



    def init_sam(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")
         # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


        sam2_checkpoint = "/home/frida/Projects/hairbot/scripts/utils/sam2_config/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)

        self.sam = SAM2ImagePredictor(sam2_model)

    def show(self):
        """Display the RGB image from the camera for 5 seconds."""
        for _ in range(300):  # 30 FPS for 5 seconds
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No color frame available")
                continue
            
            # Get the color image
            color_image = np.asanyarray(color_frame.get_data())
            
            # Display the image
            cv2.imshow('RGB Image', color_image)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Close the OpenCV window
        cv2.destroyAllWindows()

    def get_fingers(self, visualize=False):
        """Capture and return a colored point cloud."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("issues with depth or color")
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        #segment out the fingers
        input_point = np.array([[166, 340], [168, 154], [154, 254], [65,372], [95, 136], [409,300]])
        input_label = np.array([1,1,0,0, 0,0])
        self.sam.set_image(color_image[:,:,[2,1,0]])
        masks, scores, logits = self.sam.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]


        #Best mask
        mask = masks[0].astype(bool)

        # Get intrinsic parameters
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        # Convert depth image to point cloud
        height, width = depth_image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x.flatten() - cx) / fx
        y = (y.flatten() - cy) / fy
        z = depth_image.flatten() / 1000.0

        # Filter points by mask
        flat_mask = mask.flatten()
        x = x[flat_mask]
        y = y[flat_mask]
        z = z[flat_mask]

        points = np.vstack((x * z, y * z, z)).T
        points = points[np.isfinite(points).all(axis=1)]

        # Map color to the corresponding depth points
        colors = color_image.reshape(-1, 3)[flat_mask] / 255.0
        colors = colors[np.isfinite(z)]
        # Convert BGR to RGB
        colors = colors[:, [2, 1, 0]]

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # using bounding box to crop out the pcd
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, 0.0), max_bound=(1, 1, 2.5))
        pcd = pcd.crop(bbox)
        #remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        pcd = pcd.select_by_index(ind)
        #show pcd 
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        if visualize:
            from utils.sam2_config.util_sam_vis import show_masks
            show_masks(color_image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
            o3d.visualization.draw_geometries([pcd, axes])
        return pcd
    
    
    def get_depth_image(self):
        """Capture and return a depth image."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print("issues with depth")
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image
    
    
    
    
    def get_fingers_depth_image(self, remove_outliers = True, vis = False):
        """Capture and return a depth image with the background removed using SAM."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("issues with depth or color")
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        


        # Segment out the fingers (or any other object of interest) ORINIGAL 
        # input_point = np.array([[166, 340], [168, 154], [154, 254], [65, 372], [95, 136], [409, 300]])
        # input_label = np.array([1, 1, 0, 0, 0, 0])
        
        #input_point = np.array([[188,229],[151,172], [169,174],[172,348],[170,95],[205,154], [197,340], [207,350], [201,169], [168,290], [33,223],[23,277]])
        #input_label = np.array([0,0,0,0,0, 1,1,1,1,0,0,0])
        input_point = np.array([[188,229],[151,172], [169,174],[172,348],[170,95],[205,154], [197,340], [207,350], [201,169], [168,290]])
        input_label = np.array([0,0,0,0,0, 1,1,1,1,0])
        self.sam.set_image(color_image[:, :, [2, 1, 0]])
        masks, scores, logits = self.sam.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        # Best mask
        mask = masks[0].astype(bool)
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        if vis:
            #visualize_depth_image_blues(depth_image)
            cv2.imshow("Color Image", color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            
            faded_rgb = visualize_mask_on_image(color_image, mask, mask_color=(0, 255, 0), alpha=0.5, fade_factor=0.3)
            cv2.imshow("Masked Image", faded_rgb)
            cv2.waitKey(0)
            cv2.destroyAllWindows
            #visualize mask 
        

        # Apply mask to depth image
        masked_depth_image = np.zeros_like(depth_image)
        masked_depth_image[mask] = depth_image[mask]

        if remove_outliers:
            masked_depth_image = remove_depth_outliers(masked_depth_image)


        return masked_depth_image
    
    



    def get_pcd(self):
        """Capture and return a colored point cloud."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("issues with depth or color")
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get intrinsic parameters
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        # Convert depth image to point cloud
        height, width = depth_image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x.flatten() - cx) / fx
        y = (y.flatten() - cy) / fy
        z = depth_image.flatten() / 1000.0

        points = np.vstack((x * z, y * z, z)).T
        points = points[np.isfinite(points).all(axis=1)]

        # Map color to the corresponding depth points
        colors = color_image.reshape(-1, 3) / 255.0
        colors = colors[np.isfinite(z)]
        #colors bgr to rgb
        colors = colors[:, [2, 1, 0]]
        
        # Convert to Open3D point cloud
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def find_aruco_marker_transformation(self, marker_id=2, marker_size=0.07):
        """Detect the ArUco marker and return its transformation matrix (rotation and translation)."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

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

    def get_rgb_image(self):
        """Capture and return an RGB image."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def find_aruco_marker_transformation_hand(self, marker_id=3, marker_size=0.035):
        """Detect the ArUco marker and return its transformation matrix (rotation and translation)."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

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
            print("tvecs", tvecs)
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


    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()

# Example usage:
if __name__ == "__main__":
    # Get the serial numbers of connected devices
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    # Ensure we have two devices connected
    if len(serials) != 2:
        print("Error: Exactly two devices are required.")
    else:
        # Initialize cameras
        camera_1 = Camera(serials[0])
        camera_2 = Camera(serials[1])

        try:
            # Capture colored point clouds
            pcd_1 = camera_1.get_pcd()
            pcd_2 = camera_2.get_pcd()
            T = camera_2.find_aruco_marker_transformation_hand()
            print(T)
            if pcd_1 and pcd_2:
                # Save the point clouds to PCD files
                o3d.io.write_point_cloud("pointcloud_1.pcd", pcd_1)
                o3d.io.write_point_cloud("pointcloud_2.pcd", pcd_2)
                print("Colored point clouds saved to 'pointcloud_1.pcd' and 'pointcloud_2.pcd'")
            else:
                print("Failed to capture point clouds.")
        finally:
            # Stop the cameras
            camera_1.stop()
            camera_2.stop()
