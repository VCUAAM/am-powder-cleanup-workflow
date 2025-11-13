# path_planning
**Grid-Based and Offset Path Generation for Robotic Coverage**

## 🔍 Purpose
`path_planning` generates robot-compatible coverage paths based on binary masks and geometric constraints.  
It converts 2D navigation space into ordered 3D waypoints using the aligned XYZ map.

## ⚙️ Key Functions
- Grid computation and clustering of navigable space  
- Obstacle contour detection and offset generation  
- Multi-phase raster path traversal (bottom, near side, top, far side)  
- Offset polygon expansion for complete surface coverage  
- Path export and visualization  

## 📥 Inputs / 📤 Outputs
**Inputs:**  
- `.npz` file with `mask`, `rgb`, and `xyz` arrays  

**Outputs:**  
- `robot_path.npz`: ordered 3D (x, y, z) coordinates  
- `path_overlay.png`: visualization of the generated path  

## 🧩 Notes
- Fully modular — can run standalone using `compute_path()`  
- Cluster size, smoothing, and offset parameters are user-configurable  
- Designed to maintain continuous, non-intersecting toolpaths