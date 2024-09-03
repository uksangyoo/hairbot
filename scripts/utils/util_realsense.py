import pyrealsense2 as rs
import numpy as np
import open3d as o3d

class Camera:
    def __init__(self, serial):
        """Initialize the camera with the given serial number."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(self.config)

    def get_pcd(self):
        """Capture and return a colored point cloud."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
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

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

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
