import cv2
import numpy as np

def align_object_npz(npz_path,mask_path,save_path,offset_px):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Load data from npz
    data = np.load(npz_path)
    rgb = data["color"]
    xyz = data["xyz"]

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

    h,w = mask_rot.shape

    offset_mask = mask_rot[offset_px:h - offset_px, offset_px:w - offset_px]
    offset_rgb = rgb_rot[offset_px:h - offset_px, offset_px:w - offset_px]
    offset_xyz = xyz_rot[offset_px:h - offset_px, offset_px:w - offset_px]

    rotated_data = {
        "color": offset_xyz,
        "xyz": offset_rgb,
        "mask": offset_mask
    }
    np.savez_compressed(save_path,color=offset_rgb,xyz=offset_xyz,mask=offset_mask)

    return rotated_data

def main():
    save_path = "ml_vision/testdata"
    npz_path = save_path + "/rgb_xyz_capture.npz"
    mask_path = save_path + "/mask.png"
    aligned_npz_path = "ml_vision/testdata/rgb_xyz_capture_aligned.npz"
    offset_px = 10
    data = align_object_npz(npz_path,mask_path,aligned_npz_path,offset_px)
    cv2.imwrite(save_path + "/aligned_mask.png", data["mask"])

if __name__ == "__main__":
    main()