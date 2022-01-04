from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import torch

from pathlib import Path
import collections
import numbers
import random
import os


class BirdDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.x = []
        self.y = []
        self.transform = transform

        if mode == "train":
            labels = open(os.path.join(self.root_dir, 'train.txt'))

        elif mode == 'eval':
            labels = open(os.path.join(self.root_dir, 'val.txt'))

        for label in labels:
            label_list = label.split(',')
            self.x.append(label_list[0])
            self.y.append(int(label_list[1]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image_path = self.x[index]
        image = Image.open(image_path).convert('RGB')
        image = image.copy()

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


def Dataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


def _random_colour_space(x):
    output = x.convert("HSV")
    return output


class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift

    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)

        return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)


def make_dataset(mode, data_root, img_size):
    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))

    transform = [
        transforms.RandomAffine(degrees=30, shear=50, fillcolor=0),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(
            distortion_scale=0.5, p=0.5, fill=0),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        RandomShift(3),
        transforms.RandomApply([colour_transform]),
    ]

    data_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomApply(transform, p=0.5),
        transforms.RandomApply([transforms.RandomRotation(
            (-90, 90), expand=False, center=None)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    data_transform_dev = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if (mode == "train"):
        data_set = BirdDataset(data_root, mode, data_transform_train)
    elif (mode == "eval"):
        data_set = BirdDataset(data_root, mode, data_transform_dev)
    elif (mode == "test"):
        data_set = BirdDataset(data_root, mode, data_transform_test)

    return data_set
