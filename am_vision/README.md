# RealSenseCamera.py

The `RealSenseCamera` class provides a complete, configurable interface for acquiring synchronized **RGB-D (color + depth)** data from an **Intel RealSense D435i** camera. It manages stream initialization, exposure control, white balance, filtering, point cloud conversion, and optional debugging visualizations.  

This class is used to generate aligned RGB, depth, and XYZ data for downstream machine vision or robot perception modules.

I generally found it more helpful to go to Google rather than trying to use the documentation on the Intel GitHub, but you do you

---

## Key Features
- **Automatic RealSense pipeline setup** — initializes aligned color and depth streams at 848×480 resolution, 60 Hz.  
- **Custom auto-exposure algorithm** — replaces the built-in exposure logic for more consistent brightness control.  
- **Configurable white balance and brightness targets** — ensures consistent lighting for downstream edge detection or ML tasks.  
- **Spatial, temporal, and hole-filling filters** — smooth and stabilize depth data while filling small gaps.  
- **Decimation and threshold filters** — reduce noise and limit sensing range to a maximum distance of 2 m.  
- **Point cloud generation** — converts aligned frames into XYZ coordinates matched with RGB pixels.  
- **Data export** — saves compressed `.npz` datasets (color + xyz) and optionally `.ply` point clouds for debugging.  

---

## Initialization

```python
from realsense_camera import RealSenseCamera

camera = RealSenseCamera()
```
Upon creation, the class:
1. Configures and starts the RealSense pipeline.  
2. Initializes alignment, filtering, and point cloud utilities.  
3. Sets image resolution (848×480) and frame rate (60 fps).  

---

## Core Methods

| Method | Description |
|--------|--------------|
| `configure_image()` | Applies exposure, emitter, and white balance settings prior to capture. |
| `get_bright_diff(exposure)` | Evaluates average brightness relative to the target goal for auto-exposure calibration. |
| `calibrate_exposure()` | Runs an iterative exposure-tuning loop using mean grayscale intensity feedback. |
| `normalize_cmap(img)` | Applies a JET colormap to a depth image for visualization. |
| `convert_to_xyz(pc)` | Converts RealSense point cloud output into a structured XYZ image aligned with RGB. |
| `write_ply(xyz, rgb)` | Exports a `.ply` point cloud for external visualization or validation. |
| `capture()` | Executes a full acquisition cycle — filtering, alignment, conversion, and saving to `.npz`. |
| `reset()` | Resets the RealSense device in case of a runtime or USB communication error. |

---

## Example Usage

```python
camera = RealSenseCamera()
camera.capture()
```
After capture, an aligned dataset is stored in the working directory as:
```
data/rgb_xyz_capture.npz
```
Optionally, when debugging is enabled, the class also saves:
```
am_vision/data/raw_rgb.png
am_vision/data/raw_depth.png
am_vision/data/xyz.ply
```

---

## Auto-Exposure and White Balance

- `auto_exposure = True` enables a custom brightness-based control loop.  
- The algorithm iteratively adjusts exposure to achieve a target grayscale intensity (`brightness_goal = 180`).  
- White balance may be automatic (`auto_wb = True`) or fixed (`auto_wb = False`, with `self.wb` defined manually).  

These settings ensure consistent illumination across captures, especially in environments where depth sensing is sensitive to lighting changes.

---

## Filtering and Depth Processing Pipeline

Each capture applies the following RealSense filters in sequence:
1. **Decimation filter** — reduces image resolution to stabilize data.  
2. **Threshold filter** — clips measurements beyond `max_distance = 2 m`.  
3. **Disparity transform (depth → disparity)** — optimizes for spatial filtering.  
4. **Spatial filter** — smooths and fills small holes in depth data.  
5. **Temporal filter** — averages across 15 frames to remove noise.  
6. **Disparity transform (disparity → depth)** — restores depth domain.  
7. **Hole filling** — fills remaining small gaps in depth frames.  

The resulting frames are aligned to the color stream before XYZ conversion.

If you are curious about the post-processing filters, read [here](https://dev.realsenseai.com/docs/post-processing-filters)

---

## Safety and Debugging Behavior

When `self.debugging = True`, additional logging and outputs are produced:
- Saves intermediate raw RGB and depth colormap images.  
- Writes a `.ply` file containing the captured point cloud.  
- Prints time per capture, exposure calibration steps, and brightness diagnostics.  
- Displays success or failure messages during exposure calibration.  

The camera will automatically **reset** if a runtime error or USB disconnection occurs, then retry capture.

---

## Dependencies

- `pyrealsense2`  
- `numpy`  
- `opencv-python`  
- `tqdm`  
- `time`  
