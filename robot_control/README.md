# UR5Robot.py

The `UR5Robot` class provides a high-level Python interface for controlling a **UR5e collaborative robot** via the **RTDE (Real-Time Data Exchange)** protocol. It handles network connectivity, TCP (Tool Center Point) configuration, motion execution, and frame conversions between the robot base, camera, and vacuum tool frames.

This class is designed for **autonomous motion tasks** such as scanning, probing, or additive-manufacturing inspection.

For extended usage with the `ur-rtde` package, please [read the documentation](sdurobotics.gitlab.io/ur_rtde)

One note - the `pip install` method for the `ur-rtde` package does not work with every version of Python. Check wheel compatability or build from source. 

---

## Key Features
- **Automatic connection management** — verifies socket and RTDE connections with GUI prompts if unavailable.  
- **Motion control** — supports `moveJ` and `moveL` commands in both joint and Cartesian space, with asynchronous execution support.  
- **Path utilities** — generates robot paths from Cartesian waypoints with tunable velocity, acceleration, and blending.  
- **TCP configuration** — provides predefined TCP offsets for the base, camera, and vacuum end-effectors.  
- **Frame transformation** — converts RGB-D data from the camera frame to the robot base frame using stored extrinsic calibration.  
- **Safety testing mode** — reduces velocity and adds a protective Z-offset to prevent collisions during offline testing.  

---

## Initialization

```python
from ur5_robot import UR5Robot

robot = UR5Robot()
```

Upon initialization, the class will:
1. Verify socket connectivity to the robot (default IP: `192.168.1.102`).
2. Establish RTDE control and receive interfaces.
3. Prompt the user to enable *Remote Control Mode* on the teach pendant if not already active.

---

## Core Methods

| Method | Description |
|--------|--------------|
| `check_testing()` | Displays a warning if testing mode is active. |
| `socket_check()` | Verifies robot network connection before RTDE initialization. |
| `steady_check()` | Waits until the robot completes its current motion. |
| `get_path(cart_path, ori)` | Builds a moveable path from Cartesian waypoints and target orientation. |
| `move_path(npz)` | Loads and executes a saved motion path (`.npz`) from the `data/` directory. |
| `moveJ(pose, abs_j=True)` | Executes a MoveJ command — joint-space if `abs_j=True`, otherwise computes IK. |
| `moveL(pose, abs_j=True)` | Executes a MoveL command — Cartesian motion, optionally with inverse kinematics. |
| `get_tcp()` | Returns the current TCP pose `[x, y, z, rx, ry, rz]`. |
| `print_pos()` | Prints the current joint positions to the console. |
| `set_tcp(frame)` | Sets the current TCP frame; valid options: `'base'`, `'camera'`, `'vacuum'`. |
| `convert_to_base(name)` | Converts stored RGB-D data from camera to robot base frame using extrinsic calibration. |
| `shutdown(reset=True)` | Safely stops all RTDE processes and closes network connections. |

---

## Example Usage

```python
robot = UR5Robot()
robot.set_tcp('vacuum')
robot.move_path('path_example.npz')
robot.shutdown()
```

---

## Calibration and Frame Alignment

The following extrinsic parameters between the RealSense depth and color sensors are embedded in the class and were obtained using:

```bash
rs-enumerate-devices -c
```

```python
self.d2c_R  # 3×3 rotation matrix
self.d2c_T  # translation vector
```

These are applied during `convert_to_base()` to transform 3D pointcloud data from the camera frame into the UR5e base coordinate frame. The method also applies known TCP offsets and a small z-offset correction for accurate tool-to-surface alignment.

---

## Safety, Testing, and Debugging Behavior

When `self.testing = True`, the following safeguards are applied:
- **Velocity** is reduced to *0.05 m/s* to minimize impact risk.  
- **Tool height** is offset upward by *+55 mm* to prevent collisions with the test bed.  
- A **GUI warning** appears before execution to remind the operator of testing mode.  

When `self.debugging = True`, the following debugging features are activated:
- Real-time printing of TCP poses, motion paths, and pointcloud transformation outputs.  
- Human-readable formatting for joint and Cartesian data to assist with calibration and tuning.  
- Optional printouts of path data in millimeters for alignment verification.  

Ensure **Remote Control Mode** is active on the UR5e teach pendant and that the robot is connected to the `VCU_UR5e` network before running any commands.

---

## Dependencies

- `ur-rtde`
- `numpy`
- `rtde_control`
- `rtde_receive`
- `scipy`
- `tkinter`
- `socket`
- `time`
