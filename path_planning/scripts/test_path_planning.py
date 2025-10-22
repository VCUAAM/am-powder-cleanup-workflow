import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from math import *

# Computing clustered grid based on cluster size
def compute_grid(mask,offset_px,cluster_size):
    # Apply offset to mask
    kernel_size = offset_px*2 + 1
    mask_offset = cv2.erode(mask, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)

    # Cluster mask
    h, w = mask_offset.shape
    clusters_h = h // cluster_size
    clusters_w = w // cluster_size

    # Create clustered grid from mask
    cluster_grid = np.zeros((clusters_h, clusters_w), dtype=np.uint8)
    for i in range(clusters_h):
        for j in range(clusters_w):
            block = mask_offset[i*cluster_size:(i+1)*cluster_size, j*cluster_size:(j+1)*cluster_size]
            if np.any(block > 0):
                cluster_grid[i,j] = 1

    indices = np.argwhere(cluster_grid == 1)

    # Get the first occurrence
    if indices.size > 0:
        start = tuple(indices[0])
        print(f"First index of 1: {start}")
    else:
        print(f"1 not found in the array.")
    
    return start, cluster_grid

# Finding all points that are reachable for a given cluster
def compute_reachable(start, grid):
    h, w = grid.shape
    reachable = set()
    q = deque([start])
    while q:
        i, j = q.popleft()
        if (i, j) in reachable or grid[i, j] == 0:
            continue
        reachable.add((i, j))
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < h and 0 <= nj < w:
                q.append((ni, nj))
    return reachable

def generate_spiral_coords(grid):
    """
    Generate coordinates following a right-turn spiral pattern
    that fills the entire grid boundary inward, ignoring interior obstacles.
    - Only restricted by grid borders (0 ≤ i < h, 0 ≤ j < w)
    - Always turn right if forward is blocked or already visited
    """
    h, w = grid.shape
    visited = set()
    spiral_coords = []

    # Directions: up, right, down, left
    dirs = [(-1,0), (0,1), (1,0), (0,-1)]
    dir_idx = 1  # start moving right

    # Start at top-left corner of the grid (not necessarily white)
    current = (0, 0)
    spiral_coords.append(current)

    while True:
        visited.add(current)
        ci, cj = current
        moved = False

        # Try to move forward; if blocked or visited, turn right
        for _ in range(4):  # one full rotation max
            di, dj = dirs[dir_idx]
            ni, nj = ci + di, cj + dj

            # Only stay inside grid bounds
            if (0 <= ni < h and 0 <= nj < w and (ni, nj) not in visited):
                # Move forward
                current = (ni, nj)
                spiral_coords.append(current)
                moved = True
                break
            else:
                # Turn right
                dir_idx = (dir_idx + 1) % 4

        if not moved:
            break
    spiral_plot = np.asarray(spiral_coords)
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    plt.plot(spiral_plot[:,1],spiral_plot[:,0])
    ax.set_xlim(-1, np.max(spiral_plot[:,1]) + 1)  # Assuming a lower limit of 0 for x
    ax.set_ylim(-1, np.max(spiral_plot[:,0]) + 1)
    plt.savefig('path_planning/scripts/testdata/spiral_grid.png')

    return spiral_coords

# Function to choose new target when stuck
def choose_new_target(neighbors, reachable, visited, current):
    ci, cj = current
    if not neighbors or all(n in visited for n in neighbors):
        # If you're stuck or only have visited neighbors,
        # find a new unvisited reachable point to aim for
        unvisited = [p for p in reachable if p not in visited]

        if unvisited:
            # Choose the closest unvisited reachable point
            target = min(
                unvisited,
                key=lambda p: abs(p[0] - ci) + abs(p[1] - cj)
            )
            return target
        else:
            # All points visited or blocked
            return None
        
# A* search over clusters using spiral heuristic
def compute_path(reachable, grid, start = (0,0),cluster_size = 10):
    """
    A* search over clusters using spiral heuristic.
    Heuristic now directly follows spiral order, preventing back-and-forth on horizontals.
    Each cluster = single point, already visited clusters are never revisited.
    """
    h, w = grid.shape
    center = (h//2, w//2)
    visited = set()
    path_px = []

    # Precompute spiral order list of cluster coordinates
    spiral_coords = generate_spiral_coords(grid)

    #return np.asarray(spiral_coords)
    current = start
    while len(visited) < len(reachable):
        if current not in visited:
            visited.add(current)
        ci, cj = current
        center_y = ci*cluster_size + cluster_size//2
        center_x = cj*cluster_size + cluster_size//2
        path_px.append((center_y, center_x))
        
        # Simple A* step to move 1 cluster toward target (4-connected)
        if spiral_coords.index(current) + 1 >= len(spiral_coords):
            break
        i = 1
        while True:
            if spiral_coords[spiral_coords.index(current) - 1] in reachable and spiral_coords[spiral_coords.index(current) - 1] not in visited:
                target = spiral_coords[spiral_coords.index(current) - 1]
                break
            target = spiral_coords[spiral_coords.index(current) + i]
            if grid[target] == 1:
                break
            i += 1
        ti, tj = target

        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] == 1:
                neighbors.append((ni, nj))
        kj = 0
        if [x for x in neighbors if x not in visited] == []:
            target = choose_new_target(neighbors,reachable,visited,current)
            if target:
                ti, tj = target
                kj = 1
            else:
                if len(visited) == len(reachable):
                    print('finished up')
                    break
                else:
                    print('something broke')
                    quit()
        if (5,3) in visited:
            print('idk wtf man')
        print('not retargeted')
        print(target,current)
        if kj == 1:
            print('retargeted')
            print(target,current)
            print(neighbors)
            #quit()
        # Heuristic:
        # 1. Prefer closer to target (Manhattan distance)
        # 2. Penalize revisits heavily (+1000 per revisit)
        # 3. Small bias for continuing straight (optional if you track prev_dir)
        def neighbor_cost(x):
            dist_to_target = abs(x[0] - ti) + abs(x[1] - tj)
            revisit_penalty = 1000 if x in visited else 0
            impossible_penalty = 100000 if x not in reachable else 0

            return dist_to_target + revisit_penalty + impossible_penalty

        if target in neighbors:
            current = target
        else:
            current = min(neighbors, key=neighbor_cost)
        if kj == 1:
            print(current)
            print('fucky wucky')

    return np.asarray(path_px)



def main():
    # Parameters
    save_path = "path_planning/scripts/testdata"
    npz_file = save_path + "/rgb_xyz_capture_aligned.npz"
    output_file = save_path + "/robot_path.npz"
    cluster_size = 10 # pixels
    offset_px = 5 # pixels

    # Load data
    data = np.load(npz_file)
    mask = data['mask'].astype(np.uint8)  # 0=obstacle, 255=free
    # rgb = data['rgb']  # Uncomment if RGB is valid
    xyz = data['xyz']
    start, cluster_grid = compute_grid(mask, offset_px, cluster_size)
    
    # Save clustered grid image for debugging purposes
    grid_fig = np.asarray(cluster_grid*255)
    cv2.imwrite(save_path + '/clustered_grid.png',grid_fig)

    reachable = compute_reachable(start,cluster_grid)

    path_px = compute_path(reachable,cluster_grid,start,cluster_size)

    #path_xyz = np.zeros((len(path_pixels),3), dtype=np.float32)
    #for idx,(y,x) in enumerate(path_pixels):
        #path_xyz[idx] = xyz[y,x]

    # =====================
    # SAVE PATH
    # =====================
    #np.savez_compressed(output_file, path=path_xyz)
    #print(f"Saved robot path with {len(path_xyz)} points to {output_file}")

    fig,ax = plt.subplots()
    plt.plot(path_px[:,0],path_px[:,1],)
    plt.savefig(save_path + '/path_overlay.png')

if __name__ == "__main__":
    main()