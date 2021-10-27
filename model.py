from pytorch_grad_cam import GradCAM
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy
import random
import matplotlib.pyplot as plt
import itertools

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# 1. Get all the classes and the all the paths to images in dataset
classes = []
image_paths = []

for data_path in glob.glob('dataset' + '/*'):
    classes.append(data_path.split('/')[-1])
    image_paths.append(glob.glob(data_path + '/*'))

image_paths = list(itertools.chain.from_iterable(image_paths))
random.shuffle(image_paths)

print('Image Path: ', image_paths[0])
print('Class example: ', classes[0])

# 2. Split the dataset into train and test (80:20)
train_image_paths, test_image_paths = image_paths[:int(0.8*len(image_paths))], image_paths[int(0.8*len(image_paths)):] 

