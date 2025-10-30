import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Visualizer for the path
def visualizer(path,boundaries):
    fig,ax = plt.subplots()
    plt.plot(path[:,0],path[:,1],)
    rect = patches.Rectangle((boundaries[2],boundaries[0]),boundaries[3] - boundaries[2],boundaries[1] - boundaries[0],linewidth=1,edgecolor='black',facecolor='black')
    ax.add_patch(rect)
    plt.savefig('C:/Users/schorrl/Documents/GitHub/vcu_am_post_processing/path_planning/scripts/testdata/path_overlay.png')

# Computing clustered grid based on cluster size
def compute_grid(offset_mask,cluster_size = 10):
    h,w = offset_mask.shape 
    clusters_h = h // cluster_size
    clusters_w = w // cluster_size

    inverted = cv2.bitwise_not(offset_mask)
    mask_contour,_ = cv2.findContours(inverted,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(mask_contour,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest_contour)
    bounded = offset_mask.copy()
    bounded[y:y+h,x:x+w] = 0

    # Create clustered grid from mask
    cluster_grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)
    for i in range(clusters_h):
        for j in range(clusters_w):
            block = bounded[i*cluster_size:(i + 1)*cluster_size, j*cluster_size:(j + 1)*cluster_size]
            if np.any(block > 0):
                cluster_grid[i,j] = 1

    return cluster_grid

def path_planner(h,w,boundaries):
    path = []
    bottom, top = boundaries[0],boundaries[1]
    left,right = boundaries[2],boundaries[3]

    ###########################
    # Phase 1: Bottom section #
    ###########################
    
    # Simply raster back and forth until the top of the obstacle is met
    for j in range(bottom):
        if j % 2 == 0:
            path.append((0,j))
            path.append((w,j))
        else: 
            path.append((w,j))
            path.append((0,j))
    
    #####################################
    # Phase 2: Wrapping around obstacle #
    #####################################

    # Move row by row across bottom of obstacle
    if abs(path[-1][0] - 0) > abs(path[-1][0] - w): 
        upp = w
        low = right
        dir_adj = 1
    else:
        upp = left
        low = 0
        dir_adj = 0

    for j in range(path[-1][1] + 1, top + 1):
        if j % 2 == 0:
            path.append((low + dir_adj,j))
            path.append((upp - 1 + dir_adj,j))
        else:
            path.append((upp - 1 + dir_adj,j))
            path.append((low + dir_adj,j))
    
    ##########################
    # Phase 3: Moving to top #
    ##########################

    # Checking to see if path is already at width bounds
    if path[-1][0] != 0 and abs(path[-1][0] - w) > abs(path[-1][0] - 0):
        path.append((0,path[-1][1]))
    elif path[-1][0] != w and abs(path[-1][0] - 0) > abs(path[-1][0] - w):
        path.append((w,path[-1][1]))
    
    # Adding path to height boundary
    path.append((path[-1][0],h))

    ########################
    # Phase 4: Top section #
    ########################

    # Setting direction and bounds
    if path[-1][0] == w: 
        dir_adj = 1
    else:
        dir_adj = 0

    upp = w - dir_adj
    low = 1 - dir_adj

    # Adding points to the path
    for j in range(h,top,-1):
        if (h - j) % 2 == 1 - dir_adj:
            path.append((upp,j))
            path.append((low,j))
        else:
            path.append((low,j))
            path.append((upp,j))

    # Correcting for ending on wrong side of image
    if abs(path[-1][0] - 0) == 1:
        path.append((w,path[-1][1]))
    if abs(path[-1][0] - w) == 1:
        path.append((0,path[-1][1]))
    
    #######################################
    # Phase 5: Other side wrapped section #
    #######################################

    # Setting direction and bounds 
    if abs(path[-1][0] - 0) > abs(path[-1][0] - w): 
        upp = w
        low = right
        dir_adj = 1
    else:
        upp = left - 1
        low = 0
        dir_adj = 0

    # Adding points to the path
    for j in range(top, bottom - 1, -1):
        if (j - top) % 2 == dir_adj:
            path.append((low + dir_adj,j))
            path.append((upp,j))
        else:
            path.append((upp,j))
            path.append((low + dir_adj,j))
    
    return path

# Helper function to faciliatate path planning
def compute_path(offset_mask, cluster_size,visualize = False):
    grid = compute_grid(offset_mask,cluster_size)
    h, w = grid.shape
    path_px = []

    # Find the obstacle bounds
    ys, xs = np.where(grid == 0)

    # Performing a simple raster scan if no obstacle is detected
    if len(xs) == 0 or len(ys) == 0:
        path = []
        for i in range(h):
            row = [(i, j) for j in (range(w) if i % 2 == 0 else range(w - 1, -1, -1))]
            path.extend(row)
    
    # If obstacle is detected, perform complex scan
    else:
        boundaries = [ys.min(),ys.max(),xs.min(), xs.max()]
        path = path_planner(h,w,boundaries)

    for (i,j) in path:
        center_y = i*cluster_size + cluster_size//2
        center_x = j*cluster_size + cluster_size//2
        path_px.append((center_y, center_x))

    if visualize:
        visualizer(np.asarray(path),boundaries)

    return np.asarray(path_px)

def main():
    
    # Parameters
    save_path = "path_planning/scripts/testdata"
    npz_file = save_path + "/rgb_xyz_capture_aligned.npz"
    output_file = save_path + "/robot_path.npz"
    cluster_size = 10 # pixels

    # Load data
    data = np.load(npz_file)
    mask = data['mask'].astype(np.uint8)
    xyz = data['xyz']
    
    path_px = compute_path(mask,cluster_size,visualize=True)
    print(path_px)

if __name__ == "__main__":
    main()