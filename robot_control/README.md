# robot_control
**Robot Interface and Execution Layer**

## 🔍 Purpose
`robot` serves as the interface between generated paths and the physical robot system.  
It converts path coordinates into motion commands, applies calibration transforms, and manages execution safety.

## ⚙️ Key Functions
- Robot frame calibration and coordinate transformation  
- Trajectory interpolation and smoothing  
- Command streaming for robot motion (ABB, UR, etc.)  
- Logging and safety monitoring  

## 📥 Inputs / 📤 Outputs
**Inputs:**  
- `robot_path.npz` from `path_planner`  
- Robot configuration or calibration file  

**Outputs:**  
- Streamed motion commands to robot controller  
- Execution logs and optional visualization  

## 🧩 Notes
- Built for flexible integration with any e-series Universal Robot
- Designed to maintain synchronization between planned and executed motion
