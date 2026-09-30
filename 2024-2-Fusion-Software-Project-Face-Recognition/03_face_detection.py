"""Portfolio reference code extracted from the original Colab notebook.
External datasets and local paths are intentionally omitted.
"""

import cv2
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# ? ?? (Haar Cascade Classifier ?)
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# ? ???? ?
def train_with_face_detection_haar(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    best_acc = 0.0
    loss_dict = {"train": [], "val": []}
    acc_dict = {"train": [], "val": []}

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"{model_name}_best_model.pth")

            if phase == 'val':
                scheduler.step(epoch_loss)

    print(f'Best val Acc: {best_acc:.4f}')
    torch.save(loss_dict, f"{model_name}_loss_dict.pth")
    torch.save(acc_dict, f"{model_name}_acc_dict.pth")
    return model

# ResNet50  ?
model_5 = models.resnet50(pretrained=True)
model_5.fc = nn.Linear(model_5.fc.in_features, n_features)  # ????  ??
model_5.to(device)

optimizer_5 = torch.optim.Adam(model_5.parameters(), lr=1e-5)
scheduler_5 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_5, mode='min', factor=0.1, patience=10, verbose=True)

train_with_face_detection_haar(model_5, 'resnet50_haar', loss_function, optimizer_5, scheduler_5, max_epochs, dataloaders, dataset_sizes, device)


import cv2
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# ? ?? (Haar Cascade Classifier ?)
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB -> Grayscale???n    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# InceptionV3 ? ? ?? ?
def inception_train_with_face_detection_haar(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')
        print('-' * 10)

        # ?epoch ???? ?
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # ???
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                #  ??n                optimizer.zero_grad()

                with autocast():  # Mixed precision training ???n                    # ? ?? ?
                    for i in range(inputs.size(0)):  #  ? ?n                        img = inputs[i].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        img = (img * 255).astype('uint8')  # ??0-255 ????OpenCV?  ????n                        faces = detect_faces_haar(img)
                        # ? ????? ??????? (?? ? ??????)
                        # ??? ???? ? ??.

                    # ??? ?
                    outputs = model(inputs)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs  # outputs ???? [0]???
                    _, preds = torch.max(logits, 1)  # logits???? ??
                    loss = loss_function(logits, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # ?????  ???n            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # ???? ??
        scheduler.step(epoch_loss)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# model_7 ? (InceptionV3 )
model_7 = models.inception_v3(pretrained=True)
model_7.AuxLogits.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_7.AuxLogits.fc.in_features, n_features)
)
model_7.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_7.fc.in_features, n_features)
)
model_7.to(device)

# ???? transform ?
inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3??? ?
    transforms.ToTensor(),          # ????n    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ???n])

# ?? ?
inception_train_dataset = OpenCVDataset(train_data_dir, transform=inception_transform)
inception_val_dataset = OpenCVDataset(val_data_dir, transform=inception_transform)
inception_test_dataset = OpenCVDataset(test_data_dir, transform=inception_transform)

# ?????
batch_size = 16
inception_dataloaders = {
    'train': torch.utils.data.DataLoader(inception_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(inception_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(inception_test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ??? ?? ?
optimizer_7 = torch.optim.Adam(model_7.parameters(), lr=1e-5)
scheduler_7 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_7, mode='min', factor=0.1, patience=10, verbose=True)

#  ?
inception_train_with_face_detection_haar(model_7, 'inception_v3_haar', loss_function, optimizer_7, scheduler_7, max_epochs, inception_dataloaders, dataset_sizes, device)


