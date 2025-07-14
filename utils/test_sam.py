import cv2
import numpy as np
from util_realsense import Camera, visualize_depth_image_blues
# Initialize the camera (replace 'serial_number' with your actual RealSense device serial)
camera = Camera('128422271985')
camera.init_sam()
# Capture the depth image
masked_depth_image = camera.get_fingers_depth_image(remove_outliers=True, vis= True)
# Check if a valid depth image was captured
if masked_depth_image is not None:
    
    masked_depth_image[masked_depth_image >3000] = 0
    mean_depth = np.mean(masked_depth_image[masked_depth_image > 5])
    print("mean" , np.mean(masked_depth_image))
    # Normalize the depth image for visualization (optional)
    depth_visual =cv2.normalize(masked_depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = np.uint8(depth_visual)
   
    # Display the depth image
    cv2.imshow("Masked Depth Image", depth_visual)
    
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()
    
    visualize_depth_image_blues(depth_visual)
    np.save("depth_image.npy", depth_visual)
else:
    print("Failed to capture depth image.")
