import cv2
import numpy as np
import math
import random

width, height = 500, 500
canvas = np.zeros((height, width, 3), dtype="uint8")

# --- 1. Randomize square size by ±10% ---
scale_x = random.uniform(0.9, 1.1)
scale_y = random.uniform(0.9, 1.1)

rect_w = int(300 * scale_x)   # Original 400-100 = 300
rect_h = int(300 * scale_y)

# Centered placement
center = [width // 2, height // 2]
x1 = center[0] - rect_w // 2
y1 = center[1] - rect_h // 2
x2 = center[0] + rect_w // 2
y2 = center[1] + rect_h // 2

# --- 2. Draw rounded rectangle (white fill) ---
radius = 15

# Create a mask for the rounded shape
mask = np.zeros((height, width), dtype="uint8")

# Draw filled center rectangle
cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

# Draw 4 filled quarter-circles for corners
cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)

# Apply white color to the mask region
canvas[mask == 255] = (255, 255, 255)

# --- 3. Draw the black star polygon ---
outer_radius = 50
inner_radius = 20
points = []

for i in range(10):
    angle = i * math.pi / 5
    r = outer_radius if i % 2 == 0 else inner_radius
    x = int(center[0] + r * math.cos(angle - math.pi/2))
    y = int(center[1] + r * math.sin(angle - math.pi/2))
    points.append([x, y])

points = np.array([points], dtype=np.int32)
cv2.fillPoly(canvas, points, (0, 0, 0))

# --- 4. Rotate whole canvas by 15 degrees ---
angle_r = 15
(h, w) = canvas.shape[:2]
M = cv2.getRotationMatrix2D(center, angle_r, 1.0)
cos = abs(M[0, 0])
sin = abs(M[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

rotated = cv2.warpAffine(canvas, M, (new_w, new_h), borderValue=(0, 0, 0))

print(f"Square size: {rect_w}x{rect_h}")
# --- 5. Display and save ---
cv2.imwrite("ml_vision/test/test_mask.png", rotated)