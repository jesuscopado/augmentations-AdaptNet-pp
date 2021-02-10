import argparse
import math
import os
import random

import cv2
import numpy as np
from skimage import transform
from tqdm import tqdm

'''

** Notes **

From the paper Self-Supervised Model Adaptation for Multimodal Semantic Segmentation:

The augmentations that we apply include rotation (-13° to 13°),
skewing (0.05 to 0.10), scaling (0.5 to 2.0), vignetting (210
to 300), cropping (0.8 to 0.9), brightness modulation (-40
to 40), contrast modulation (0.5 to 1.5) and flipping.

----

From their github:

Augment the training data(images and gt_mapped). In our work, we first resized the images in the dataset to 768x384 pixels and then apply a series of augmentations (random_flip, random_scale and random_crop).

Convert the training data (augmented), test data and validation data into the .tfrecords format. Create a .txt file for each set having entries in the following format:

   path_to_modality1/0.png path_to_label/0.png
   path_to_modality1/1.png path_to_label/1.png
   path_to_modality1/2.png path_to_label/2.png

'''

freiburg_forest_colors = np.array([
    [170, 170, 170],  # Road
    [0, 255, 0],  # Grass
    [102, 102, 51],  # Vegetation
    [0, 60, 0],  # Tree
    [0, 120, 255],  # Sky
    [0, 0, 0],  # Obstacle
]).astype(np.uint8)  # RGB representation of the classes


def rotate(image, gt, limits=(-13, 13)):
    angle = random.uniform(limits[0], limits[1])

    # Apply rotation to image
    image = (transform.rotate(image, angle=angle, mode='edge') * 255).astype(np.uint8)
    gt = (transform.rotate(gt, angle=angle, mode='edge') * 255).astype(np.uint8)
    return image, gt


def skew(image, gt, limits=(-0.05, 0.10)):
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=random.uniform(limits[0], limits[1]))

    # Apply transform to image data
    image = (transform.warp(image, inverse_map=afine_tf, mode='edge') * 255).astype(np.uint8)
    gt = (transform.warp(gt, inverse_map=afine_tf, mode='edge') * 255).astype(np.uint8)
    return image, gt


'''
# NOT USED
def scale(image, limits=(0.5, 2.0)):
    # Apply scaling (zoom) to image

    # TODO: decide what to do here
    #  Scaling 0.5 to 1.0 will shrink the image, how to fill all other pixels
    #  Scaling >1.0 will need a crop, should it be random or central?

    return transform.rescale(image, scale=random.uniform(limits[0], limits[1]), mode='constant', multichannel=True)
'''


def crop(image, gt, limits=(0.8, 1.0)):
    height, width = image.shape[:2]

    # Randomly generate an inverse zoom factor and find the cropped dimensions
    inverse_zoom_factor = random.uniform(limits[0], limits[1])
    crop_height = round(inverse_zoom_factor * height)
    crop_width = round(inverse_zoom_factor * width)

    # Randomly find the corner (x, y) from which to crop
    max_x = width - crop_width
    max_y = height - crop_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Crop the images
    cropped_image = image[y: y + crop_height, x: x + crop_width]
    cropped_gt = gt[y: y + crop_height, x: x + crop_width]

    # Resize back to the original dimension
    return cv2.resize(cropped_image, (width, height)), cv2.resize(cropped_gt, (width, height))


def vignette(image, limits=(210, 300)):
    height, width = image.shape[:2]

    # Generate vignette kernel with two gaussian kernels
    sigma = random.uniform(limits[0], limits[1])
    kernel_gauss_x = cv2.getGaussianKernel(width, sigma)
    kernel_gauss_y = cv2.getGaussianKernel(height, sigma)
    kernel = kernel_gauss_y * kernel_gauss_x.T

    # Create mask with the kernel
    mask = (255 * kernel / np.linalg.norm(kernel))

    # Apply mask on the image
    masked = (image * np.expand_dims(mask, axis=-1)).astype(np.uint8)

    # Attenuate the effect by getting an image in between the original and the masked
    factor = random.uniform(0.3, 1.0)
    output = (factor * image + (1 - factor) * masked).astype(np.uint8)

    return output


def modulate_brightness(image, limits=(-40, 40)):
    # # Convert to HSV
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # # Add the random brightness value
    # h, s, v = cv2.split(hsv)
    # v = np.clip(cv2.add(v, random.randint(limits[0], limits[1])), 0, 255)
    # final_hsv = cv2.merge((h, s, v))
    #
    # # Convert back to RGB
    # return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(image, alpha=1, beta=random.randint(limits[0], limits[1]))


def modulate_contrast(image, limits=(0.5, 1.5)):
    return cv2.convertScaleAbs(image, alpha=random.uniform(limits[0], limits[1]), beta=0)


def flip(image, gt):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        gt = cv2.flip(gt, 1)
    return image, gt


def fix_label_colors(gt, color_palette=freiburg_forest_colors):
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    def closest_color(color):
        min_distance = math.inf
        closest = np.array([0, 0, 0]).astype(np.uint8)
        for color_label in color_palette:
            distance = np.sum(np.abs(color_label - color))
            if distance < min_distance:
                min_distance = distance
                closest = color_label
        return closest

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            color = gt[i, j, :].copy()
            found = False
            k = 0
            while not found and k < color_palette.shape[0]:
                if np.array_equal(color, color_palette[k]):
                    found = True
                k += 1
            if not found:
                gt[i, j, :] = closest_color(color)

    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    return gt


def main(input_dir, output_dir, aug_number):
    images_subfolder_name = 'rgb'
    gts_subfolder_name = 'GT_color'

    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, images_subfolder_name))
        os.makedirs(os.path.join(output_dir, gts_subfolder_name))

    # Create txt file with the entry pairs
    f = open('entry_pairs.txt', 'w')

    # Iterate over all images in the folder
    for filename_w_ext in tqdm(os.listdir(os.path.join(input_dir, images_subfolder_name))):

        # Extract image filename from path
        filename, _ = os.path.splitext(filename_w_ext)
        gt_filename_w_ext = filename.split('_')[0] + '_mask.png'
        gt_filename, _ = os.path.splitext(gt_filename_w_ext)

        # Construct path to both image and gt
        image_path = os.path.join(os.path.join(input_dir, images_subfolder_name), filename_w_ext)
        gt_image_path = os.path.join(os.path.join(input_dir, gts_subfolder_name), gt_filename_w_ext)

        # Read image and ground truth image
        image_src = cv2.imread(image_path)
        gt_src = cv2.imread(gt_image_path)

        # Augment the image
        for i in range(aug_number):
            image, gt = image_src.copy(), gt_src.copy()

            # cv2.imshow('original', image)
            # cv2.waitKey()

            image, gt = rotate(image, gt)
            image, gt = skew(image, gt)
            image, gt = crop(image, gt)
            image, gt = flip(image, gt)
            image = vignette(image)
            image = modulate_brightness(image)
            image = modulate_contrast(image)

            # Fix label colors that have changed because of the transformations
            # gt = fix_label_colors(gt)

            # cv2.imshow('augmented', image)
            # cv2.waitKey()

            # Save augmented images on disk and their paths in the entry pair txt file
            path_to_image = os.path.join(os.path.join(output_dir, images_subfolder_name), f'{filename}_{i}.jpg')
            path_to_gt = os.path.join(os.path.join(output_dir, gts_subfolder_name), f'{gt_filename}_{i}.png')
            cv2.imwrite(path_to_image, image)
            cv2.imwrite(path_to_gt, gt)
            f.write(f'{path_to_image} {path_to_gt}\n')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment image.')
    parser.add_argument('--input_dir', default='img', help='dir where both subdirs with images and labels are')
    parser.add_argument('--output_dir', default='augmented', help='dir in which to save the augmentations')
    parser.add_argument('--aug_number', default=10, help='number of times the image will be augmented')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.aug_number)