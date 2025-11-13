# ml_vision
**Machine Learning and Object Detection for Additive Manufacturing**

## 🔍 Purpose
`ml_vision` provides model inference tools for detecting and distinguishing key geometric elements (e.g., build and feed cylinders) from RGB imagery.

## ⚙️ Key Functions
- YOLOv5-based model loading and inference  
- Bounding box extraction and filtering  
- Automatic differentiation between feed and build cylinders  
- Bounding-box refinement and expansion for orientation correction  

## 📥 Inputs / 📤 Outputs
**Inputs:**  
- RGB image data or `.npz` file with `rgb` field  
- Pretrained YOLOv5 model weights  

**Outputs:**  
- Detected bounding boxes for key components  
- Coordinates and classes exported as `.json` or `.txt`  
- Optional mask or region proposals for downstream processing  

## 🧩 Notes
- Designed to interface directly with `am_vision` preprocessing outputs  
- Model tuning parameters are configurable through YAML  
- Supports oriented bounding box approximation via Hough-based postprocessing
