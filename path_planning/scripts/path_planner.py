import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Visualizer for the path
def visualizer(path,polygon):
    fig,ax = plt.subplots()
    polygon = np.squeeze(polygon)
    smooth_poly = cv2.approxPolyDP(polygon, 2.5, True).reshape(-1, 2)
    plt.plot(path[:,0],path[:,1],)
    patch = MplPolygon(smooth_poly, closed=True, facecolor='black', edgecolor='black')
    ax.add_patch(patch)
    plt.savefig('C:/Users/schorrl/Documents/GitHub/vcu_am_post_processing/path_planning/scripts/testdata/path_overlay.png')

def offset_polygon(polygon, offset_distance, smooth=True, epsilon_ratio=0.01):
    """
    Expand a polygon uniformly outward by a given number of pixels and optionally smooth/approximate it.

    Parameters:
        polygon (np.ndarray): Polygon as an Nx1x2 or Nx2 array of integer coordinates.
        offset_distance (int): Number of pixels to expand outward.
        smooth (bool): Whether to smooth and approximate the polygon.
        epsilon_ratio (float): Approximation ratio (fraction of contour perimeter).

    Returns:
        np.ndarray: Offset polygon as integer coordinates (same shape as OpenCV contours Nx1x2).
    """
    # Ensure Nx2 format
    poly = np.squeeze(polygon).astype(np.float32)
    centroid = np.mean(poly, axis=0)

    # Compute outward offset
    directions = poly - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_dirs = directions / norms
    offset_poly = poly + unit_dirs * offset_distance

    # Round outward
    diffs = offset_poly - poly
    offset_poly_rounded = np.where(diffs > 0, np.ceil(offset_poly), np.floor(offset_poly)).astype(int)

    # --- optional smoothing and polygon approximation ---
    if smooth:
        # Convert to binary mask temporarily for smoothing
        x_min, y_min = offset_poly_rounded[:,0].min(), offset_poly_rounded[:,1].min()
        x_max, y_max = offset_poly_rounded[:,0].max(), offset_poly_rounded[:,1].max()
        mask = np.zeros((y_max - y_min + 20, x_max - x_min + 20), dtype=np.uint8)
        shifted = offset_poly_rounded - [x_min - 10, y_min - 10]
        cv2.fillPoly(mask, [shifted], 255)

        # Apply Gaussian blur and threshold to smooth jagged edges
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Extract smoothed contour
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)

        # Polygon simplification (epsilon controls smoothing strength)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon_ratio * peri, True)

        # Shift back to original coordinate frame
        offset_poly_rounded = approx + [x_min - 10, y_min - 10]
    offset_poly = np.squeeze(offset_poly_rounded, axis=1)

    return offset_poly

def generate_offset_polygons(largest_contour, offset_step=10):
    """
    Repeatedly create outward offset polygons until none remain inside
    the bounding rectangle of the original largest contour.

    Parameters:
        largest_contour (np.ndarray): Base contour from cv2.findContours.
        start (tuple): (x, y) coordinate to align polygon start order.
        offset_step (int): Distance (in pixels) for each outward offset.

    Returns:
        list[np.ndarray]: List of polygons, each of shape (N+1, 2),
                          ordered and closed (start==end).
    """
    # Compute bounding rectangle of the original contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = (x, y, x + w, y + h)

    # Storage for offset polygons
    polygons = []
    iteration = 1
    current_poly = np.squeeze(largest_contour, axis=1)

    while True:
        # Compute next offset polygon
        offset_poly = offset_polygon(current_poly, offset_step)
        #offset_poly = np.squeeze(offset_poly, axis=1)

        # Check whether all points are outside bounding box
        xs, ys = offset_poly[:, 0], offset_poly[:, 1]
        inside_x = (xs >= bbox[0]) & (xs <= bbox[2])
        inside_y = (ys >= bbox[1]) & (ys <= bbox[3])
        inside_mask = inside_x & inside_y

        # If all points are outside, stop
        if not np.any(inside_mask):
            break

        # Append to list and continue
        polygons.append(offset_poly)
        current_poly = offset_poly.copy()
        iteration += 1

    return polygons

# Computing clustered grid based on cluster size
def compute_grid(mask, cluster_size=10):
    h, w = mask.shape
    clusters_h = h // cluster_size
    clusters_w = w // cluster_size

    inverted = cv2.bitwise_not(mask)
    mask_contour, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(mask_contour, key=cv2.contourArea)

    # Bounding rectangle for termination check
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounded = mask.copy()
    bounded[y:y+h, x:x+w] = 0

    # Convert to cluster grid (same as your current logic)
    cluster_grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)
    for i in range(clusters_h):
        for j in range(clusters_w):
            block = bounded[i*cluster_size:(i + 1)*cluster_size, j*cluster_size:(j + 1)*cluster_size]
            if np.any(block > 0):
                cluster_grid[i, j] = 1

    return cluster_grid, largest_contour

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

def reorder_polygon_start(poly, start):
    """
    Reorder polygon points so that the vertex closest to `start`
    becomes the first point, preserving sequence order.

    Parameters:
        poly (np.ndarray): (N, 2) array of polygon vertices.
        start (tuple): (x, y) coordinates of the starting point.

    Returns:
        np.ndarray: Reordered polygon of shape (N, 2).
    """
    # Compute Euclidean distance to the start point
    dists = np.linalg.norm(poly - np.array(start), axis=1)
    
    # Find index of the closest vertex
    idx = np.argmin(dists)

    # Roll the array so that the closest point is first
    poly_reordered = np.roll(poly, -idx, axis=0)
    poly_closed = np.vstack([poly_reordered, poly_reordered[0]])
    
    return poly_closed

# Helper function to faciliatate path planning
def compute_path(mask, cluster_size,visualize = False):
    grid,polygon = compute_grid(mask,cluster_size)
    h_m,w_m = mask.shape
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
        center_y = center_y - 1 if center_y == h_m else center_y
        center_x = center_x - 1 if center_x == w_m else center_x
        path_px.append((center_y, center_x))
    
    # Generate concentric offset polygons
    offset_polygons = generate_offset_polygons(polygon, cluster_size)

    for layer in offset_polygons:
        poly = reorder_polygon_start(layer,path_px[-1])
        for [i,j] in poly:
            path_px.append((i,j))
            path.append((i/cluster_size,j/cluster_size))

    if visualize:
        visualizer(np.asarray(path_px),polygon)

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

    robot_path = []
    for [i,j] in path_px:
        robot_path.append(xyz[i][j])
    
if __name__ == "__main__":
    main()