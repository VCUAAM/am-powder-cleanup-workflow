# VCU Additive Manufacturing Post-Processing
This project is the graduate research of Logan Schorr. The goal of this research is to determine a path to automate post-processing of additive manufacturing, processes such as heat treatment, support removal, and polishing. 

The current implementation of is on a [Universal Robots UR5e](https://www.universal-robots.com/products/ur5-robot/), with a [Robotiq 2F-140](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) end effector with custom grippers, utilizing a [Vzense DCAM560CPro](https://www.vzense.com/ptof/dcam560c) 3D camera for vision capabilities. It is primarily programmed in [Python3](https://docs.python.org/3/tutorial/), using [ROS2 Rolling](https://docs.ros.org/en/rolling/index.html). This respository is broken into **4** packages.

## My documentation is SUPER outdated, but the following lines is accurate albeit temporary until I actually have the time to write something better
  - AM Vision - depreciated, look here if you want but it's just where I keep some old scripts that I can steal some code from
  - ml_vision
      - This is where all of the current image processing happens. Inside this directly is where I cloned the model directory. The goal of this module is to identify the location of the build cylinder and provide the coordinates for the path planning.
  - Path planning
      - This is where path planning occurs, with the processed image. Generates contours around any printed objects
  - robot_control
      - Should be pretty self explanatory. Controls a robot
 
## stuff to fix later when i have time

- move j5 of robot to align image to square with camera
- make sure that all walls are approximately equal to "verify" bounding box
- move robot automatically if entire build cylinder is not in image