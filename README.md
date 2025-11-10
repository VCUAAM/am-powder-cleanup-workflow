# VCU Additive Manufacturing Post-Processing
This project is the graduate research of Logan Schorr. The goal of this research is to determine a path to automate post-processing of additive manufacturing, processes such as heat treatment, support removal, and polishing. 

The current implementation of is on a [Universal Robots UR5e](https://www.universal-robots.com/products/ur5-robot/), with a [Robotiq 2F-140](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) end effector with custom grippers, utilizing a [Vzense DCAM560CPro](https://www.vzense.com/ptof/dcam560c) 3D camera for vision capabilities. It is primarily programmed in [Python3](https://docs.python.org/3/tutorial/), using [ROS2 Rolling](https://docs.ros.org/en/rolling/index.html). This respository is broken into **4** packages.
  - AM Vision
  - UR Path Planning
  - VCU UR Driver
  - AM Post-Processing Interfaces
 


mention that in order to use any of the machine learning aspects, yolov5 needs to be installed locally inside ml_vision


maybe features? 

- checking average white value of image after taking picture to adjust exposure value to make thresholding more consistent
- move j5 of robot to align image to square with camera
- make sure that all walls are approximately equal to "verify" bounding box
- move robot automatically if entire build cylinder is not in image
- need to correct matrix by color to depth offset extrinsic matrix