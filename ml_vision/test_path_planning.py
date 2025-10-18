import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =====================
# PARAMETERS
# =====================
save_path = "ml_vision/testdata"
npz_file = save_path + "/rgb_xyz_capture_aligned.npz"
output_file = save_path + "/robot_path.npz"
cluster_size = 10
offset_pixels = 5
offset_meters = 0.015
visualize = True

# =====================
# LOAD DATA
# =====================
data = np.load(npz_file)
mask = data['mask'].astype(np.uint8)  # 0=obstacle, 255=free
# rgb = data['rgb']  # Uncomment if RGB is valid
xyz = data['xyz']

# =====================
# APPLY OFFSET
# =====================
kernel_size = offset_pixels*2 + 1
mask_offset = cv2.erode(mask, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)

# =====================
# CLUSTER MASK
# =====================
h, w = mask_offset.shape
clusters_h = h // cluster_size
clusters_w = w // cluster_size

cluster_grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)
for i in range(clusters_h):
    for j in range(clusters_w):
        block = mask_offset[i*cluster_size:(i+1)*cluster_size, j*cluster_size:(j+1)*cluster_size]
        if np.any(block > 0):
            cluster_grid[i,j] = 1

indices = np.argwhere(cluster_grid == 1)

# Get the first occurrence
if indices.size > 0:
    first_index = indices[0]
    print(f"First index of 1: {first_index}")
else:
    print(f"1 not found in the array.")
    
# =====================
# SPIRAL HEURISTIC
# =====================
def spiral_order_indices(h, w):
    """Generate a 2D array of spiral indices from outside-in"""
    spiral = np.zeros((h, w), dtype=int)
    val = 0
    top, left = 0, 0
    bottom, right = h-1, w-1
    while top <= bottom and left <= right:
        for j in range(left, right+1):
            spiral[top,j] = val; val+=1
        top+=1
        for i in range(top, bottom+1):
            spiral[i,right] = val; val+=1
        right-=1
        for j in range(right, left-1, -1):
            spiral[bottom,j] = val; val+=1
        bottom-=1
        for i in range(bottom, top-1, -1):
            spiral[i,left] = val; val+=1
        left+=1
    return spiral

# =====================
# A* IMPLEMENTATION
# =====================
def astar_clusters(grid, spiral_grid, start=(0,0)):
    """
    A* search over clusters using spiral heuristic.
    Heuristic now directly follows spiral order, preventing back-and-forth on horizontals.
    Each cluster = single point, already visited clusters are never revisited.
    """
    h, w = grid.shape
    visited = set()
    path_pixels = []

    # Precompute spiral order list of cluster coordinates
    spiral_coords = sorted([(i,j) for i in range(h) for j in range(w) if grid[i,j]==1],
                           key=lambda x: spiral_grid[x])
    spiral_idx = 0  # index of the next target in spiral_coords

    current = start
    while spiral_idx < len(spiral_coords):
        # If current cluster is free and not visited, add to path
        if current not in visited and grid[current]:
            ci, cj = current
            center_y = ci*cluster_size + cluster_size//2
            center_x = cj*cluster_size + cluster_size//2
            path_pixels.append((center_y, center_x))
            visited.add(current)

        # Determine next target along spiral
        while spiral_idx < len(spiral_coords) and spiral_coords[spiral_idx] in visited:
            spiral_idx += 1
        if spiral_idx >= len(spiral_coords):
            break
        target = spiral_coords[spiral_idx]

        # Simple A* step to move 1 cluster toward target (4-connected)
        ci, cj = current
        ti, tj = target
        neighbors = []
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = ci+di, cj+dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni,nj]==1 and (ni,nj) not in visited:
                neighbors.append((ni,nj))
        if not neighbors:
            # No available neighbors, jump directly to target (isolated clusters)
            current = target
        else:
            # Choose neighbor closest to target (manhattan distance)
            current = min(neighbors, key=lambda x: abs(x[0]-ti)+abs(x[1]-tj))

    return path_pixels



spiral_grid = spiral_order_indices(clusters_h, clusters_w)
path_pixels = np.asarray(astar_clusters(cluster_grid, spiral_grid, start=(15,23)))
# =====================
# MAP PIXELS TO XYZ
# =====================
#path_xyz = np.zeros((len(path_pixels),3), dtype=np.float32)
#for idx,(y,x) in enumerate(path_pixels):
    #path_xyz[idx] = xyz[y,x]

# =====================
# SAVE PATH
# =====================
#np.savez_compressed(output_file, path=path_xyz)
#print(f"Saved robot path with {len(path_xyz)} points to {output_file}")

# =====================
# OPTIONAL VISUALIZATION
# =====================

fig,ax = plt.subplots()
plt.plot(path_pixels[:,0],path_pixels[:,1],)
#rect = Rectangle((x1,y1),x2 - x1,y2 - y1,linewidth=1, edgecolor='black', facecolor='none')
#ax.add_patch(rect)
#ax.axis('off')
plt.savefig(save_path + '/path_overlay.png')
#plt.show()
#mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#for y,x in path_pixels:
    #mask_vis[y,x] = [0,0,255]
#plt.savefig(save_path + '/path_overlay.png')
#plt.imshow(mask_vis)
#plt.title("A* Spiral Path Overlay")
#plt.show()
