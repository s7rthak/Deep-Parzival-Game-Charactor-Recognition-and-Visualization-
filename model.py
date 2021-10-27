from pytorch_grad_cam import GradCAM
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from data_augmentation import *
import time
import copy
import torch

train_transforms = A.Compose(
    [
        # A.SmallestMaxSize(max_size=350),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        # A.RandomCrop(height=256, width=256),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        # A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        # A.SmallestMaxSize(max_size=350),
        # A.CenterCrop(height=256, width=256),
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

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}



NO_AUGMENTATION = 0
TO_GRAY = 1
TO_BLURRED = 2
TO_OUTLINES = 3
TO_SKETCH = 4

class GameCharacterDataset(Dataset):
    def __init__(self, image_paths, transform=None, augmentation=NO_AUGMENTATION):
        self.image_paths = image_paths
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file_path = self.image_paths[idx]
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

        label = image_file_path.split('/')[-2]
        label = class_to_idx[label]

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

        return image, label



train_dataset = GameCharacterDataset(train_image_paths, train_transforms)
test_dataset = GameCharacterDataset(test_image_paths, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(idx_to_class[preds[j].item()]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

MODELS = {
    'resnet50': models.resnet50(pretrained=True),
    'effNetB0': EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(classes))
}
model_name = 'effNetB0'
model_ft = MODELS[model_name]
print(model_ft)

# Work as fixed feature extractor?
FIXED_FEATURE_EXTRACTOR = False
if FIXED_FEATURE_EXTRACTOR:
    for param in model_ft.parameters():
        param.requires_grad = False

if not model_name.startswith('effNet'):
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

visualize_model(model_ft)