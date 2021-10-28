import cv2
import numpy as np
from torch.utils.data import Dataset
from data_augmentation import *

NO_AUGMENTATION = 0
TO_GRAY = 1
TO_BLURRED = 2
TO_OUTLINES = 3
TO_SKETCH = 4

class GameCharacterDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=None, augmentation=NO_AUGMENTATION):
        self.image_paths = image_paths
        self.transform = transform
        self.augmentation = augmentation
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file_path = self.image_paths[idx]
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

        input = {}
        input["original"] = np.copy(image)
        input["class"] = image_file_path.split('/')[-2]
        input["filename"] = image_file_path

        label = image_file_path.split('/')[-2]
        label = self.class_to_idx[label]

        if self.augmentation == TO_GRAY:
            image = to_gray(image)
        elif self.augmentation == TO_BLURRED:
            image = to_blurred(image)
        elif self.augmentation == TO_OUTLINES:
            image = to_outlines(image)
        elif self.augmentation == TO_SKETCH:
            image = to_sketch(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        input["tensor"] = image

        return input, label