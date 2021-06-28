import cv2
import numpy as np

freiburg_forest_colors = np.array([
    [170, 170, 170],  # Road
    [0, 255, 0],  # Grass
    [102, 102, 51],  # Vegetation
    [0, 60, 0],  # Tree
    [0, 120, 255],  # Sky
    [0, 0, 0],  # Obstacle
]).astype(np.uint8)  # RGB representation of the classes

gt_path = 'augmented/GT_color/b1-99445_mask_3.png'
gt = cv2.imread(gt_path, cv2.IMREAD_ANYCOLOR)

gt_colors = gt.reshape(-1, gt.shape[-1])
unique_colors = np.unique(gt_colors, axis=0)

print(unique_colors)
