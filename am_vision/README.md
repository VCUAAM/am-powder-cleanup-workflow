# am_vision
**Additive Manufacturing Vision Preprocessing**

## 🔍 Purpose
`am_vision` provides low-level image and depth preprocessing tools for additive manufacturing data captured by the Intel RealSense D435i.  
It handles thresholding, alignment, and mask generation prior to higher-level model inference.

> ⚠️ **Note:** This package is currently in a **rough, pre-refactor state**. It contains legacy scripts from earlier versions of the project and will be restructured to separate vision utilities, calibration logic, and preprocessing workflows.

## ⚙️ Key Functions
- Image and depth thresholding for isolating build regions  
- Rotation and alignment of RGB/XYZ arrays  
- Mask and bounding-box generation  
- Utility functions for image normalization and cropping  

## 📥 Inputs / 📤 Outputs
**Inputs:**  
- Aligned `.npz` files containing `rgb`, `xyz`, and `depth` arrays  

**Outputs:**  
- Binary masks highlighting build regions  
- Oriented and cropped RGB/XYZ images  

## 🧩 Notes
Future updates will integrate:
- Unified preprocessing pipeline  
- Depth-based segmentation and surface filtering  
- Improved alignment for non-planar builds
