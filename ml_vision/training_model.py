import torch

model = torch.hub.load('ml_vision\\yolov5','custom',path='ml_vision\\yolov5\\runs\\train\\exp7\\weights\\best.pt',source='local',force_reload=True)

img = 'ml_vision\\training_data\\square\\rgb\\square_009.png'

# Perform inference
results = model(img)

# Display results (optional)
#results.show()
detections = results.pandas().xyxy[0] # Bounding box coordinates, class, confidence
print(detections)