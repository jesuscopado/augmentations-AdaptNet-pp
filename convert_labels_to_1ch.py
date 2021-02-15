import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='augmented/GT_color')
args = parser.parse_args()

rgb_to_label_id = {
    (170, 170, 170):    1,  # Road
    (0, 255, 0):        2,  # Grass
    (102, 102, 51):     3,  # Vegetation
    (0, 60, 0):         3,  # Tree
    (0, 120, 255):      4,  # Sky
    (0, 0, 0):          5,  # Obstacle
}  # Freiburg forest mapping from RBG to class id

for filename in tqdm(os.listdir(args.input_dir)):
    image_path = os.path.join(args.input_dir, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = np.zeros(img.shape[:2], dtype=np.uint8)
    for k, v in rgb_to_label_id.items():
        out[(img == k).all(axis=2)] = v
    cv2.imwrite(image_path, out)
