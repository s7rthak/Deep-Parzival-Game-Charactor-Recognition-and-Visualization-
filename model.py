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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

cam_dict = {'ezio': [3, 8, 12, 19, 36], 'geralt': [1, 2, 16, 136], 'mario': [35, 39, 40, 74], 'pacman': [1, 4, 8, 32, 147], 'pikachu': [15, 43, 44, 70, 146], 'ryu': [48, 57, 69, 88], 'sonic hedgehog': [5, 9, 33]}
cam_image_paths = []
for key, value in cam_dict.items():
    for i in range(len(value)):
        cam_image_paths.append('dataset/{}/{}_{}.png'.format(key, key, value[i]))

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

# Load Dataset
train_dataset = GameCharacterDataset(train_image_paths, class_to_idx, train_transforms)
test_dataset = GameCharacterDataset(test_image_paths, class_to_idx, test_transforms)
# Dataset on which GradCAM operates.
cam_dataset = GameCharacterDataset(cam_image_paths, class_to_idx, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
cam_loader = DataLoader(cam_dataset, batch_size=1, shuffle=False)

dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
MODELS = {
    'resnet50': models.resnet50(pretrained=True),
    'effnetB0': EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(classes)),
    'vggnet16': models.vgg16(pretrained=True),
    'vggnet19': models.vgg19(pretrained=True)
}
model_name = 'resnet50'

def train_model(model, criterion, optimizer, scheduler, target_layer, tl_str, num_epochs=25):
    since = time.time()

    train_accs, val_accs = [], []
    y_true, y_pred = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    global cam
    saved_img_no = 0
    # Create directory to save visualisations.
    if not os.path.exists('visualise/{}/{}'.format(model_name, tl_str)):
        os.makedirs('visualise/{}/{}'.format(model_name, tl_str))
    # Create directory for saving miclassifications.
    if not os.path.exists('misclassified'):
        os.makedirs('misclassified')
    # Delete previous run misclassifications and store current.
    # files = glob.glob('misclassified/*')
    # for f in files:
    #     os.remove(f)

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
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_accs.append(epoch_acc.item())
            else:
                val_accs.append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    draw_plot(train_accs, val_accs, 'Accuracy', "{}_accuracy.png".format(model_name))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    model.eval()
    # Store misclassified images.
    for inputs, labels in test_loader:
        inputs_tensor = inputs["tensor"].to(device)
        labels = labels.to(device)

        outputs = model(inputs_tensor)
        _, preds = torch.max(outputs, 1)

        # Get confusion matrix entries
        y_pred.append(idx_to_class[int(preds.item())])
        y_true.append(idx_to_class[int(labels.item())])

        if (preds != labels).item():
            cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs_tensor[0], 0))
            grayscale_cam = grayscale_cam[0, :]
            rgb_image = np.float32(np.squeeze(inputs["original"].cpu().detach().numpy())) / 255
            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            filename = inputs["filename"][0].split('/')[-1]
            file, ext = filename.split('.')
            cv2.imwrite("misclassified/{}_as_{}.{}".format(file, idx_to_class[preds.item()], ext), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # Save confusion matrix
    y_true = [y_true[i] if y_true[i] != 'sonic hedgehog' else 'sonic' for i in range(len(y_true))]
    y_pred = [y_pred[i] if y_pred[i] != 'sonic hedgehog' else 'sonic' for i in range(len(y_pred))]
    sc = [classes[i] if classes[i] != 'sonic hedgehog' else 'sonic' for i in range(len(classes))]
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=sc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=sc)
    disp.plot()
    plt.savefig('confusion_matrix.png', bbox_inches='tight')

    # Run gradCAM on selected pics.
    for inputs, labels in cam_loader:
        inputs_tensor = inputs["tensor"].to(device)
        labels = labels.to(device)

        outputs = model(inputs_tensor)

        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
        grayscale_cam = cam(input_tensor=torch.unsqueeze(inputs_tensor[0], 0))
        grayscale_cam = grayscale_cam[0, :]
        rgb_image = np.float32(np.squeeze(inputs["original"].cpu().detach().numpy())) / 255
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        filename = inputs["filename"][0].split('/')[-1]
        cv2.imwrite("visualise/{}/{}/{}".format(model_name, tl_str, filename), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
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
            inputs = inputs["tensor"].to(device)
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

model_ft = MODELS[model_name]

SHOW_MODEL = True
if SHOW_MODEL:
    print(model_ft)

# Work as fixed feature extractor?
FIXED_FEATURE_EXTRACTOR = False
if FIXED_FEATURE_EXTRACTOR:
    for param in model_ft.parameters():
        param.requires_grad = False

# Add in custom layer to accomodate our dataset classes
if model_name.startswith('resnet'):
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))
elif model_name.startswith('vggnet'):
    num_ftrs = model_ft.classifier[-1].in_features
    features = list(model_ft.classifier.children())[:-1]
    features.extend([torch.nn.Linear(num_ftrs, len(classes))])
    model_ft.classifier = nn.Sequential(*features)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, model_ft.layer1[-1], 'layer1', num_epochs=5)

# visualize_model(model_ft)