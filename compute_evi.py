import os

import cv2
import numpy as np
from scipy import stats

import argparse
from tqdm import tqdm

# Constants
L = 1
C1 = 6
C2 = 7.5
G = 2.5


def compute_evi(p_nir, p_red, p_blue):
    # evi = G * (p_nir - p_red) / (p_nir + (C1 * p_red) - (C2 * p_blue) + L + np.finfo(float).eps)
    evi = G * (p_nir - p_red) / (p_nir + (C1 * p_red) - (C2 * p_blue) + L)
    # evi[evi > 1] = 1
    # evi[evi < -1] = -1
    return ((evi + 1) * 255 / 2).astype(np.uint8)


def compute_evi2(p_nir, p_red):
    evi2 = 2.5 * ((p_nir - p_red) / (p_nir + 2.4 * p_red + 1))
    return ((evi2 + 1) * 255 / 2.5).astype(np.uint8)


def minmax_norm(numpy_array):
    return (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min())


def main(input_dir):
    rgb_folder_name, nir_folder_name, nir_gray_folder_name, evi_gray_folder_name = 'rgb', 'nir', 'nir_gray', 'evi_gray'
    evi2_gray_folder_name = 'evi2_gray'

    # Create output dir if it doesn't exist
    evi2_gray_dir = os.path.join(input_dir, evi2_gray_folder_name)
    if not os.path.exists(evi2_gray_dir):
        os.makedirs(evi2_gray_dir)

    # Iterate over all images in the folder
    for filename_w_ext in tqdm(os.listdir(os.path.join(input_dir, rgb_folder_name))):
        # Extract image filename from path
        filename, _ = os.path.splitext(filename_w_ext)
        basename = filename.split('_')[0]

        # Read images and extract needed channels
        rgb_path = os.path.join(os.path.join(input_dir, rgb_folder_name), f'{basename}_Clipped.jpg')
        nir_gray_path = os.path.join(os.path.join(input_dir, nir_gray_folder_name), f'{basename}.tif')
        # evi_given = os.path.join(os.path.join(input_dir, evi_gray_folder_name), f'{basename}.tif')

        bgr = cv2.imread(rgb_path)
        p_nir = cv2.imread(nir_gray_path, cv2.IMREAD_GRAYSCALE) / 255.
        p_red = bgr[:, :, 2] / 255.
        # p_blue = bgr[:, :, 0] / 255.

        evi2 = compute_evi2(p_nir, p_red)
        # print(stats.describe(evi2))
        # cv2.imshow('EVI2', evi2)
        # cv2.waitKey()

        # Write EVI2 on disk
        evi2_path = os.path.join(evi2_gray_dir, f'{basename}.tif')
        cv2.imwrite(evi2_path, evi2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='C:/projects/datasets/freiburg_forest_annotated/train')
    args = parser.parse_args()
    main(args.input_dir)
