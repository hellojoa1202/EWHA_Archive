"""Portfolio reference code extracted from the original Colab notebook.
External datasets and local paths are intentionally omitted.
"""

from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch

# Mixed precision training ?
scaler = GradScaler()

def train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    file_name = f"{model_name}_model_params.pt"
    best_acc = 0.0
    loss_dict = {"train": [], "val": []}  # ? ?????
    acc_dict = {"train": [], "val": []}   # ????????

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # ? 
            else:
                model.eval()    # ?? 

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()     # ?? ?0? ?

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():  # Mixed precision training ???n                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_function(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc.item())

            # ? ?????n            plt.plot(range(len(loss_dict[phase])), loss_dict[phase])
            plt.title(f"{model_name} - {phase} Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f"{model_name}_{phase}_Loss.png")
            plt.close()

            # ????????n            plt.plot(range(len(acc_dict[phase])), acc_dict[phase])
            plt.title(f"{model_name} - {phase} Accuracy")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.savefig(f"{model_name}_{phase}_ACC.png")
            plt.close()

            print(f'{phase} Total Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), file_name)

            if phase == 'val':
                scheduler.step(epoch_loss)   # ?? ?

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    #  ?? ? ? ?/???
    torch.save(loss_dict, f"{model_name}_loss_dict.pth")
    torch.save(acc_dict, f"{model_name}_acc_dict.pth")


loss_function = torch.nn.CrossEntropyLoss() # loss function ?
max_epochs = 10 # epoch ???
n_features = len(class_names)

model_1 = models.resnet50(pretrained=True)
model_1.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_1.fc.in_features, n_features)
)
model_1.to(device)

optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_1, 'resnet50', loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device)

model_2 = EfficientNet.from_pretrained('efficientnet-b0').to(device)
model_2._fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_2._fc.in_features, n_features)
)
model_2.to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-5)
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_2, 'efficientnet', loss_function, optimizer_2, scheduler_2, max_epochs, dataloaders, dataset_sizes, device)

from torchvision import models

model_3 = models.mobilenet_v2(pretrained=True).to(device)
model_3.classifier[1] = nn.Linear(model_3.classifier[1].in_features, n_features)
model_3.to(device)

optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=1e-5)
scheduler_3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_3, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_3, 'mobilenetv2', loss_function, optimizer_3, scheduler_3, max_epochs, dataloaders, dataset_sizes, device)

def inception_train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
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

                with autocast():  # Mixed precision training ???n                    outputs = model(inputs)
                    # outputs[0]? logits???
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

model_4 = models.inception_v3(pretrained=True)
model_4.AuxLogits.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_4.AuxLogits.fc.in_features, n_features)
)
model_4.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_4.fc.in_features, n_features)
)
model_4.to(device)

inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3??? ?
    transforms.ToTensor(),          # ????n    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ???n])

inception_train_dataset = OpenCVDataset(train_data_dir, transform=inception_transform)  # InceptionV3?? transform ?
inception_val_dataset = OpenCVDataset(val_data_dir, transform=inception_transform)      # InceptionV3?? transform ?
inception_test_dataset = OpenCVDataset(test_data_dir, transform=inception_transform)    # InceptionV3?? transform ?

# ?????
batch_size = 16
inception_dataloaders = {
    'train': torch.utils.data.DataLoader(inception_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(inception_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(inception_test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ??? ?? ?
optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=1e-5)
scheduler_4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_4, mode='min', factor=0.1, patience=10, verbose=True)

#  ?
inception_train_model(model_4, 'inception_v3', loss_function, optimizer_4, scheduler_4, max_epochs, inception_dataloaders, dataset_sizes, device)

