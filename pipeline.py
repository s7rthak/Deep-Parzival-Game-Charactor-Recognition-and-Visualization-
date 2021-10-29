from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, numpy as np, matplotlib.pyplot as plt
import random, os, random, time, copy, itertools, glob
from dataset_class import *
import torch
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def develop_model(model, blur_sd):
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

    # 2. Split the dataset into train and test (80:20)
    train_image_paths, test_image_paths = image_paths[:int(0.8*len(image_paths))], image_paths[int(0.8*len(image_paths)):] 

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    # Load Dataset
    train_dataset = GameCharacterDataset(train_image_paths, class_to_idx, train_transforms, augmentation=TO_BLURRED, blur_dict={'kernel_sz': (7,7), 'sx': blur_sd, 'sy': blur_sd})
    test_dataset = GameCharacterDataset(test_image_paths, class_to_idx, test_transforms, augmentation=TO_BLURRED, blur_dict={'kernel_sz': (7,7), 'sx': blur_sd, 'sy': blur_sd})

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    dataloaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    ## Train Model
    num_epochs = 5
    since = time.time()

    train_accs, val_accs = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    global cam
    saved_img_no = 0

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
                inputs_tensor = inputs["tensor"].to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_tensor)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs_tensor.size(0)
                running_corrects += torch.sum(preds == labels.data)


            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_accs.append(epoch_acc)
            else:
                val_accs.append(epoch_acc)

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

MODELS = {
    'resnet50': models.resnet50(pretrained=False),
    'effNetB0': EfficientNet.from_pretrained('efficientnet-b0', num_classes=7)
}
model_name = 'resnet50'
model_ft = MODELS[model_name]
model_ft = model_ft.to(device)

model_ft = develop_model(model_ft, 20)
model_ft = develop_model(model_ft, 10)
model_ft = develop_model(model_ft, 5)
model_ft = develop_model(model_ft, 2.5)
model_ft = develop_model(model_ft, 2)
model_ft = develop_model(model_ft, 1)
model_ft = develop_model(model_ft, 0.5)
model_ft = develop_model(model_ft, 0)