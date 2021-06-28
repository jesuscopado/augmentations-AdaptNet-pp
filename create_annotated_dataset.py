import argparse
import os
import glob
import shutil

from tqdm import tqdm

list_subdirs = ['ndwi_float']


def main(dataset_dir, prepro_dir):
    gt_color_dir = os.path.join(dataset_dir, 'GT_color')

    for filename in tqdm(os.listdir(gt_color_dir)):
        image_index = filename.split('_')[0]
        for subdir in list_subdirs:
            subdir_path = os.path.join(prepro_dir, subdir)
            image_path = glob.glob(os.path.join(subdir_path, f'{image_index}*'))[0]
            shutil.copy2(image_path, os.path.join(dataset_dir, subdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='C:/projects/datasets/hungary_forest_annotated/test/')
    parser.add_argument('--preprocessed_data_dir', default='C:/projects/offroad/preprocessed_images/')
    args = parser.parse_args()
    main(args.dataset_dir, args.preprocessed_data_dir)
