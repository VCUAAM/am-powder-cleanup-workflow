import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

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

def compute_path(grid):
    h, w = grid.shape
    path,path_clean = [],[]

    # Find the obstacle bounds
    ys, xs = np.where(grid == 0)

    #print([(xs,ys) for xs,ys in zip(xs,ys)])
    if len(xs) == 0 or len(ys) == 0:
        # no obstacle: simple raster scan
        print('no obstacle detected')
        for i in range(h):
            row = [(i, j) for j in (range(w) if i % 2 == 0 else range(w - 1, -1, -1))]
            path.extend(row)
        return path

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()

    # --- Phase 1: Top section (above obstacle) ---
    for i in range(top):
        if i % 2 == 0:
            for j in range(w + 1):
                path.append((i, j))
            #path.append((i,0))
            #path.append((i,w))
        else: 
            for j in range(w, -1, -1):
                path.append((i, j))
            #path.append((i,w))
            #path.append((i,0))
    
    # --- Phase 2: Around the obstacle ---
    # Move row by row across top of obstacle
    if abs(path[-1][1] - 0) > abs(path[-1][1] - w): 
        upp = w
        low = right
        dir_adj = 1
    else:
        upp = left
        low = 0
        dir_adj = 0

    for i in range(path[-1][0] + 1, bottom + 1):
        if i % 2 == 0:
            for j in range(low + dir_adj,upp + dir_adj):
                path.append((i, j))
            #path.append((i,low + dir_adj))
            #path.append((i,upp - 1 + dir_adj))
        else:
            for j in range(upp - 1 + dir_adj,low - 1 + dir_adj, -1):
                path.append((i, j))
            #path.append((i,upp - 1 + dir_adj))
            #path.append((i,low + dir_adj))

    #############################
    # Phase 3: Moving to bottom #
    #############################

    # Checking to see if path is already at width bounds
    if path[-1][1] != 0 and abs(path[-1][1] - w) > abs(path[-1][1] - 0):
        path.append((path[-1][0],0))
    elif path[-1][1] != w and abs(path[-1][1] - 0) > abs(path[-1][1] - w):
        path.append((path[-1][0],w))
    
    # Adding path to height boundary
    for i in range(path[-1][0] + 1,h + 1):
        path.append((i,path[-1][1]))

    ###########################
    # Phase 4: Bottom section #
    ###########################

    # Setting direction and bounds
    if path[-1][1] == w: 
        dir_adj = 1
    else:
        dir_adj = 0

    upp = w - dir_adj
    low = 1 - dir_adj

    # Adding points to the path
    for i in range(h,bottom, -1):
        if (h - i) % 2 == 1 - dir_adj:
            for j in (range(upp,low - 1, -1)):
                path.append((i, j))
            #path.append((i,upp))
            #path.append((i,low))
        else:
            #path.append((i,low))
            #path.append((i,upp))
            for j in range(low,upp + 1):
                path.append((i, j))

    # Correcting for ending on wrong side of image
    if abs(path[-1][1] - 0) == 1:
        path.append((path[-1][0],w))
    if abs(path[-1][1] - w) == 1:
        path.append((path[-1][0],0))
    
    #######################################
    # Phase 5: Other side wrapped section #
    #######################################

    # Setting direction and bounds 
    if abs(path[-1][1] - 0) > abs(path[-1][1] - w): 
        upp = w
        low = right
        dir_adj = 1
    else:
        upp = left - 1
        low = 0
        dir_adj = 0

    # Adding points to the path
    for i in range(bottom, top - 1, -1):
        if (i - bottom) % 2 == dir_adj:
            for j in range(low + dir_adj,upp + 1):
                path.append((i, j))
            #path.append((i,low + dir_adj))
            #path.append((i,upp))
        else:
            for j in range(upp,low - 1 + dir_adj, -1):
                path.append((i, j))
            #path.append((i,upp))
            #path.append((i,low + dir_adj))
    
    for n,(i,j) in enumerate(path):
        if n > 0 and n < len(path) - 1 and (len(set([path[n - 1][0],path[n][0],path[n + 1][0]])) == 1 or len(set([path[n - 1][1],path[n][1],path[n + 1][1]])) == 1):
            if (path[n][0] - path[n - 1][0])*(path[n][0] - path[n + 1][0]) > 0 or (path[n][1] - path[n - 1][1])*(path[n][1] - path[n + 1][1]) > 0:
                pass
            else:
                continue
        path_clean.append((i,j))

    path = np.asarray([(j,i) for (i,j) in path_clean])
    
    return path,[top,bottom,left,right]

def main():
    
    # Parameters
    save_path = "path_planning/scripts/testdata"
    npz_file = save_path + "/rgb_xyz_capture_aligned.npz"
    output_file = save_path + "/robot_path.npz"
    cluster_size = 10 # pixels

    # Load data
    data = np.load(npz_file)
    mask = data['mask'].astype(np.uint8)  # 0=obstacle, 255=free
    # rgb = data['rgb']  # Uncomment if RGB is valid
    xyz = data['xyz']
    
    cluster_grid = compute_grid(mask, cluster_size)
    start = time.perf_counter()
    # Save clustered grid image for debugging purposes
    #grid_fig = np.asarray(cluster_grid*255)
    #cv2.imwrite(save_path + '/clustered_grid.png',grid_fig)
    
    path,loc = compute_path(cluster_grid)
    end = time.perf_counter()
    time_elapsed = (end - start)*1000
    print(f"Code execution took {time_elapsed} ms.")
    path_px = []
    
    for [i,j] in path:
        center_y = i*cluster_size + cluster_size//2
        center_x = j*cluster_size + cluster_size//2
        path_px.append((center_y, center_x))

    fig,ax = plt.subplots()
    plt.plot(path[:,0],path[:,1],)
    rect = patches.Rectangle((loc[2],loc[0]),loc[3] - loc[2],loc[1] - loc[0],linewidth=1,edgecolor='black',facecolor='black')
    ax.add_patch(rect)
    plt.savefig(save_path + '/path_overlay.png')

if __name__ == "__main__":
    main()