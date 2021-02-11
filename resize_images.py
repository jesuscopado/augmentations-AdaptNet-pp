import argparse
import os

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='augmented')
args = parser.parse_args()

desired_resolution = (768, 384)

gt_path = args.input_dir
for subdirectory in ('GT_color', 'rgb'):
    for filename in tqdm(os.listdir(os.path.join(args.input_dir, subdirectory))):
        image_path = os.path.join(os.path.join(args.input_dir, subdirectory), filename)
        img = cv2.imread(image_path)
        if subdirectory is 'rgb':
            img = cv2.resize(img, desired_resolution)
        else:
            img = cv2.resize(img, desired_resolution, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(image_path, img)
