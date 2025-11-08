import torch 
import numpy as np
import pathlib
import platform
import cv2

# Runs YOLOv5 model and extracts bounding box
def extract_model_bbox(model, img,border=3,target_classes='build_cylinder',visualize=False):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(RGB_img)
    
    if visualize:
        results.show()

    # Extracting data out of the results
    class_names = results.names
    detections = results.xyxy[0].cpu().numpy()

    # Adding all boxes belonging to target class 
    boxes = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = class_names[int(cls_id)]
        
        # Filter if specific class given
        if label not in target_classes:
            continue

        off_x = max(0, (y2 - y1 - (x2 - x1)) // 2)
        bbox = [
            int(x1 - off_x - border),
            int(y1),
            int(x2 + off_x + border),
            int(y2)
        ]

        boxes.append([label, bbox, conf])

    # Sorting so that only highest confidence with matching label is returned
    return max(boxes, key=lambda b: float(b[2]))[1]


# Uses YOLO box to threshold and identify true oriented bounding box
def get_oriented_bbox(model,img,save_path=None,border=3,targ_class='build_cylinder',visualize=False):
    
    # Grabbing coordinates out of model bounding box and clipping image
    x_lo,y_lo,x_hi,y_hi = extract_model_bbox(model,img,border,targ_class,visualize)
    bounded = img[y_lo:y_hi,x_lo:x_hi]
    
    # Thresholding image to make difference between sections more distinct
    gray = cv2.cvtColor(bounded, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    scale = 255.0 / (220 - 160) #hi - lo
    thresh = (gray_f - 160) * scale #gray_f - lo
    thresh = np.clip(thresh, 0, 255).astype(np.uint8)
    
    # Canny edge detection
    blur = cv2.GaussianBlur(thresh, (11,11), 0)
    edges = cv2.Canny(blur, 1, 70)
    
    # Detecting Hough lines from Canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=11,
                            minLineLength=150, maxLineGap=50)
    
    # Draw Hough lines
    vis_hough = bounded.copy()
    for (x1,y1,x2,y2) in lines[:,0]:
        cv2.line(vis_hough, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Adding Hough lines to points array
    pts = []
    for (x1,y1,x2,y2) in lines[:,0]:
        if edges[y1, x1] == 0 or edges[y2, x2] == 0:
            # Skip lines whose endpoints aren't on real edges
            continue
        pts.append([x1,y1])
        pts.append([x2,y2])
    
    # Drawing bounding box based off of Hough lines
    pts = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts)       # center, (w,h), angle
    box = cv2.boxPoints(rect)         # 4 rotated corners
    box = np.int32(box)
    box = box.reshape((-1,1,2))
    vis_rect = bounded.copy()
    cv2.drawContours(vis_rect, [box], 0, (255,0,0), 2)

    # Creating mask based on bounding box                 
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [box], 255)
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[y_lo:y_hi, x_lo:x_hi] = mask

    # Saving images if visualization
    if visualize and save_path:
        cv2.imwrite(save_path + 'clipped.png',bounded)
        cv2.imwrite(save_path + 'edges.png',edges)
        cv2.imwrite(save_path + 'thresh.png',thresh)
        cv2.imwrite(save_path + 'bbox.png',vis_rect)
        cv2.imwrite(save_path + 'hough.png',vis_hough)
        cv2.imwrite(save_path + 'bounded_mask.png',full_mask)
    
    return full_mask

def align_mask(mask,rgb,xyz,offset_px=10):
    # --- find principal orientation of the object ---
    ys, xs = np.nonzero(mask)
    coords = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigvecs = cv2.PCACompute(coords, mean=np.array([]))
    center = tuple(mean[0])
    angle = np.degrees(np.arctan2(eigvecs[0,1], eigvecs[0,0]))

    # Snap to closest right angle 
    if abs(angle) > 45:
        angle += -abs(angle)/angle*90

    # --- build rotation matrix about the object centroid ---
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    def warp(im, interp=cv2.INTER_LINEAR):
        return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                              flags=interp, borderMode=cv2.BORDER_CONSTANT)

    # --- rotate all aligned arrays ---
    rgb_rot  = np.dstack([warp(rgb[...,c]) for c in range(3)])
    mask_rot  = warp(mask, interp=cv2.INTER_NEAREST)
    xyz_rot   = np.dstack([warp(xyz[...,c]) for c in range(3)])

    mask_rows = ~np.all(mask_rot == 0, axis=1)  # keep rows with any nonzero
    mask_cols = ~np.all(mask_rot == 0, axis=0)  # keep cols with any nonzero

    mask_rot = mask_rot[np.ix_(mask_rows, mask_cols)]
    rgb_rot = rgb_rot[np.ix_(mask_rows, mask_cols)]
    xyz_rot = xyz_rot[np.ix_(mask_rows, mask_cols)]

    # Clip all the frames to the size of the mask for ease in path planning
    h,w = mask_rot.shape

    offset_mask = mask_rot[offset_px:h - offset_px, offset_px:w - offset_px]
    offset_rgb = rgb_rot[offset_px:h - offset_px, offset_px:w - offset_px]
    offset_xyz = xyz_rot[offset_px:h - offset_px, offset_px:w - offset_px]

    rotated_data = {
        "color": offset_xyz,
        "xyz": offset_rgb,
        "mask": offset_mask
    }

    return rotated_data

def main(model_dir,img_dir):
    # Check the operating system and set the appropriate path type
    if platform.system() == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
    model = torch.hub.load('ultralytics/yolov5','custom',path=model_dir,force_reload=True)
    data = np.load(img_dir + 'rgb_xyz_capture.npz')
    rgb = data["color"]
    xyz = data["xyz"]
    border = 3
    targ_class='build_cylinder'
    mask = get_oriented_bbox(model,rgb,img_dir,border,targ_class,visualize=True)

    offset_px = 10
    rot_data = align_mask(mask,rgb,xyz,offset_px)
    #cv2.imwrite(img_dir + "/aligned_mask.png", rot_data["mask"])
    np.savez_compressed(img_dir + 'rgb_xyz_aligned.npz',
                        color=rot_data["color"],
                        xyz=rot_data["xyz"],
                        mask=rot_data["mask"])

if __name__ == '__main__':
    model_dir = 'ml_vision/best.pt'
    img_dir = 'ml_vision/test/'
    main(model_dir,img_dir)