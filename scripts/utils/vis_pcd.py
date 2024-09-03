import open3d as o3d

def vis_pcd(pcds):
    """Visualize a list of point clouds."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for pcd in pcds:
        vis.add_geometry(pcd)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Load point clouds from files
    pcd_1 = o3d.io.read_point_cloud("pointcloud_1.pcd")
    pcd_2 = o3d.io.read_point_cloud("pointcloud_2.pcd")
    
    # Visualize the point clouds
    vis_pcd([pcd_2])