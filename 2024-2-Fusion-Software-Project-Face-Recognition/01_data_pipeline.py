"""Portfolio reference code extracted from the original Colab notebook.
External datasets and local paths are intentionally omitted.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time

import torchvision
from torchvision import datasets, models, transforms

import PIL
import matplotlib.pyplot as plt
import os
import zipfile
from PIL import Image
import pandas as pd
import numpy as np

import splitfolders

import cv2
import dlib
from torchvision.transforms import functional as F

from torchvision.models import mobilenet_v2

from vggface import VGGFace
from deepface import DeepFace
from efficientnet_pytorch import EfficientNet
from torchvision.models import mobilenet_v2

import os
import shutil
import torch
from torchvision import datasets, transforms
import splitfolders
import cv2
from torch.utils.data import Dataset
from PIL import Image  # PIL ???

# train ??? 8:2?
splitfolders.ratio(train_dir, output=split2_dir, ratio=(.8, .2), group_prefix=None)

# train, val, test ????? ?

# train ? ? ? 'train' ? ?
for class_name in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_name)
    if os.path.isdir(class_path) and class_name == 'train':
        shutil.rmtree(class_path)  # 'train' ? ??

for class_name in os.listdir(val_data_dir):
    class_path = os.path.join(val_data_dir, class_name)
    if os.path.isdir(class_path) and class_name == 'train':
        shutil.rmtree(class_path)  # 'train' ? ??

# OpenCV????? ???Dataset ????
class OpenCVDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []

        # ??  ???n        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((os.path.join(class_dir, filename), class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
# External data loading omitted for portfolio archive.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR??RGB???n        img = Image.fromarray(img)  # NumPy ??PIL ?????n
        if self.transform:
            img = self.transform(img)  # ??????

        # ??????????n        label = self.classes.index(class_name)

        return img, label

# ??????
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, shear=0.1, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ?? 
train_dataset = OpenCVDataset(train_data_dir, transform=data_transforms['train'])
val_dataset = OpenCVDataset(val_data_dir, transform=data_transforms['val'])
test_dataset = OpenCVDataset(test_data_dir, transform=data_transforms['test'])

# ?????
batch_size = 16
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ?? ?
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

# ????
class_names = train_dataset.classes

# GPU ? ????? device ?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#  ?
print(f"Train size: {dataset_sizes['train']}")
print(f"Validation size: {dataset_sizes['val']}")
print(f"Test size: {dataset_sizes['test']}")
print(f"Class names: {class_names}")
print(f"Device: {device}")


import os
import random
import cv2
import matplotlib.pyplot as plt


# ?? ? (2x5 ?)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()  # 2D ??1D???n
# ?????????? ?? 
for idx, class_name in enumerate(class_names[:10]):  # ?10??? 
    # ????? ? ?? ???n    class_dir = os.path.join(train_data_dir, class_name)
    image_files = os.listdir(class_dir)

    # ?? ????? ?
    image_path = os.path.join(class_dir, random.choice(image_files))

    # OpenCV??? ?? ?
# External data loading omitted for portfolio archive.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR? RGB???n
    # ?????? 
    axes[idx].imshow(img)
    axes[idx].axis('off')  #  ??? ?
    axes[idx].set_title(class_name)  # ???? ?

# ??  
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


# ?, ? ????? ? ?
print(f"Train dataset size: {dataset_sizes['train']}")
print(f"Validation dataset size: {dataset_sizes['val']}")
print(f"Test dataset size: {dataset_sizes['test']}")

