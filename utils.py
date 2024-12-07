# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image


def normalize_dataframe(data, num_bins=23):
    """
    Normalize dataframe to avoid bias to driving straight.

    :param data: Dataframe which is to be normalized.
    :param num_bins: Number of bins to use in angle histogram.

    :return: A Normalized dataframe.
    """
    avg_samples_per_bin = len(data['Steering Angle']) / num_bins
    hist, bins = np.histogram(data['Steering Angle'], num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) * 0.5

    # Drop random straight steering angles.
    keep_probs = []
    target = avg_samples_per_bin * .3
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.0)
        else:
            keep_probs.append(1.0 / (hist[i] / target))

    # Delete from X and y with probability 1 - keep_probs[j].
    remove_list = []
    for i in range(len(data['Steering Angle'])):
        angle = data['Steering Angle'][i]
        for j in range(num_bins):
            if angle > bins[j] and angle <= bins[j + 1]:
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)

    data = data.drop(data.index[remove_list])
    return data


def load_data(file_name, columns):
    """
    Loads in data as dataframe and sets column data type.

    :param file_name: Dataset file to read in.
    :param columns: Names of each column in dataset.

    :return: A dataframe.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Data file not found: {file_name}")
    data = pd.read_csv(file_name, names=columns, header=0)
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    data[columns[:3]] = data[columns[:3]].astype(str)
    data[columns[3:]] = data[columns[3:]].astype(float)
    data = normalize_dataframe(data)
    images = data[columns[:3]]
    angles = data[columns[3]]
    return images, angles


def resize_crop(img):
    """
    Resize and crop the image to the desired size (66, 200).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if img.ndim < 2:
        raise ValueError("Input image must have at least two dimensions.")
    img = cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)
    return img


def jitter_image(path, steering):
    """
    Open image from disk and jitters it and modifies new angle.

    :param path: Path of image.
    :param steering: Steering angle corresponding to image.

    :return: Jittered image and new steering angle.
    """
    if not os.path.exists(path.strip()):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path.strip())
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    
    rows, cols, _ = img.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange / 2
    transY = numPixels * np.random.uniform() - numPixels / 2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    steering = steering + transX / transRange * 2 * valPixels
    img = cv2.warpAffine(img, transMat, (cols, rows))
    return resize_crop(img), steering


def flip_image(path):
    """
    Flips the image.

    :param path: Path of image to flip.

    :return: A flipped image.
    """
    if not os.path.exists(path.strip()):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path.strip())
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    img = cv2.flip(img, 1)  # Flip horizontally
    return resize_crop(img)


def tint_image(path):
    """
    Applies random tint to image to simulate night time.

    :param path: Path of image to flip.

    :return: A tinted image.
    """
    if not os.path.exists(path.strip()):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path.strip())
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return resize_crop(img)


def load_image(path):
    """
    Loads an image given its path.

    :param path: Path of image to load.

    :return: An image.
    """
    if not os.path.exists(path.strip()):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path.strip())
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    return resize_crop(img)
