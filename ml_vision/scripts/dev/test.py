# Source - https://stackoverflow.com/a
# Posted by Employee, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-10, License - CC BY-SA 4.0

import open3d as o3d 

def main():
    cloud = o3d.io.read_point_cloud("ml_vision/data/debugging/xyz.ply") # Read point cloud
    o3d.visualization.draw_geometries([cloud])    # Visualize point cloud      

if __name__ == "__main__":
    main()
