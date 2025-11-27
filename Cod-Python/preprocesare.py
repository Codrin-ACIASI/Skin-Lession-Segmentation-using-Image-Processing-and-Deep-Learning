# data loader
from __future__ import print_function, division
import glob
import albumentations as A
import cv2
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#==========================dataset load==========================

class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        if len(self.label_name_list) == 0:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2], dtype=np.float32)
        if len(label_3.shape) == 3:
            label = label_3[:, :, 0].astype(np.float32)
        elif len(label_3.shape) == 2:
            label = label_3.astype(np.float32)

        if len(image.shape) == 3 and len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        elif len(image.shape) == 2 and len(label.shape) == 2:
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


#--------------------------Transforms--------------------------

class ResizeFixed(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        img = transform.resize(image, (self.size, self.size), mode='constant')
        lbl = transform.resize(label, (self.size, self.size), mode='constant', order=0, preserve_range=True)

        # Normalize label to [0,1] for BCE
        lbl = lbl.astype(np.float32)
        if lbl.max() > 1.0:
            lbl /= 255.0

        sample["image"] = img
        sample["label"] = lbl
        return sample


class RandomFlips(object):
    def __init__(self, p_h=0.5, p_v=0.2):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        if np.random.rand() < self.p_h:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        if np.random.rand() < self.p_v:
            image = np.flip(image, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        sample["image"] = image
        sample["label"] = label
        return sample


class RandomBrightnessContrast(object):
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.aug = A.RandomBrightnessContrast(brightness_limit=brightness,
                                              contrast_limit=contrast,
                                              p=p)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        aug = self.aug(image=image, mask=label)
        sample["image"] = aug["image"]
        sample["label"] = aug["mask"]
        return sample


class Normalize01(object):
    def __call__(self, sample):
        image = sample["image"].astype(np.float32)
        image = image / 255.0 if image.max() > 1.0 else image
        sample["image"] = image
        return sample


class ToTensorDict(object):
    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        # copy pentru a evita negative strides
        image_tensor = torch.from_numpy(image.copy()).float()
        label_tensor = torch.from_numpy(label.copy()).float()

        # clamp label pentru BCE între 0 și 1
        label_tensor = torch.clamp(label_tensor, 0.0, 1.0)

        sample["image"] = image_tensor
        sample["label"] = label_tensor
        return sample


class HairRemoval(object):
    def __init__(self, kernel_size=17, threshold=10, dilation_size=3, inpaint_radius=3):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.dilation_size = dilation_size
        self.inpaint_radius = inpaint_radius

    def __call__(self, sample):
        image = sample["image"]
        img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, self.threshold, 255, cv2.THRESH_BINARY)

        if self.dilation_size > 0:
            d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_size, self.dilation_size))
            hair_mask = cv2.dilate(hair_mask, d_kernel, 1)

        inpainted = cv2.inpaint(img, hair_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        sample["image"] = inpainted
        return sample


class BorderRemoval(object):
    def __init__(self, threshold=10, min_border_width=5):
        self.threshold = threshold
        self.min_border_width = min_border_width

    def __call__(self, sample):
        img = sample["image"]

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape

        def find_border(start, end, step, axis=0):
            for i in range(start, end, step):
                line = gray[i, :] if axis == 0 else gray[:, i]
                if np.mean(line < self.threshold) < 0.95:
                    return i
            return start

        top = find_border(0, h, 1, axis=0)
        bottom = find_border(h - 1, -1, -1, axis=0)
        left = find_border(0, w, 1, axis=1)
        right = find_border(w - 1, -1, -1, axis=1)

        if bottom <= top or right <= left:
            sample["image"] = img
            return sample

        cropped = img[top:bottom + 1, left:right + 1].copy()  # copy pentru siguranță
        sample["image"] = cropped
        return sample


class ComposeDict(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
