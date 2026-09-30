"""Portfolio reference code extracted from the original Colab notebook.
External datasets and local paths are intentionally omitted.
"""

import matplotlib.pyplot as plt
import torch

plt.figure(figsize=(12, 8))

# ResNet50
plt.plot(range(1, len(resnet50_acc_dict['val']) + 1), resnet50_acc_dict['val'], label='ResNet50')

# ResNet50 with Haar
plt.plot(range(1, len(resnet50_haar_acc_dict['val']) + 1), resnet50_haar_acc_dict['val'], label='ResNet50 (Haar)')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy of Models')
plt.savefig('validation_accuracy.png')
plt.show()

import random
import matplotlib.pyplot as plt
import numpy as np

# ???? ??
model_7.eval()

# ??????? ????
selected_images = []
for class_name in class_names:
    class_dir = os.path.join(test_data_dir, class_name)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    selected_images.append(random.choice(images))

# 2x5 ?? ?
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for idx, img_path in enumerate(selected_images):
    # ??  ???n# External data loading omitted for portfolio archive.
    img_tensor = data_transforms['test'](img).unsqueeze(0).to(device)

    #  ?
    with torch.no_grad():
        outputs = model_7(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # ???n    img_np = np.array(img)
    axes[idx].imshow(img_np)
    axes[idx].axis('off')
    axes[idx].set_title(f"{class_names[predicted_class]}\n{confidence:.2f}")

# ?? 
plt.tight_layout()
plt.show()


model_best=model_8

# best model : resnet152 + CELoss + RMSprop
## best model ? code

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

def train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs):
    file_name = f"{model_name}_model_params.pt"
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
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc.item())

            plt.plot(range(len(loss_dict[phase])), loss_dict[phase])
            plt.savefig(f"{phase}_Loss.png")
            plt.close()

            plt.plot(range(len(acc_dict[phase])), acc_dict[phase])
            plt.savefig(f"{phase}_ACC.png")
            plt.close()

            print(f'{phase} Total Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), file_name)

            if phase == 'val':
                scheduler.step(epoch_loss)

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    torch.save(loss_dict, f"{model_name}_loss_dict.pth")
    torch.save(acc_dict, f"{model_name}_acc_dict.pth")

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)
optimizer = torch.optim.RMSprop(model_best.parameters(), lr=1e-5) # optimizer ?
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_function_best= nn.CrossEntropyLoss()

train_model(model_best, 'RMSprop', loss_function_best, optimizer, scheduler, max_epochs)

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F

def visualize_random_predictions(model, dataloader, class_names, num_images=10):
    model.eval()
    all_inputs = []
    all_labels = []

    for inputs, labels in dataloader:
        for i in range(inputs.size()[0]):
            all_inputs.append(inputs[i])
            all_labels.append(labels[i])

    #?? ?? ?
    random_indices = random.sample(range(len(all_inputs)), num_images)
    selected_images = [all_inputs[idx] for idx in random_indices]
    selected_labels = [all_labels[idx] for idx in random_indices]

    with torch.no_grad():
        for i in range(num_images):
            inputs = selected_images[i].unsqueeze(0).to(device)
            labels = selected_labels[i].to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            fig, ax = plt.subplots(1, 2, figsize=(5, 5))
            ax[0].axis('off')
            img = inputs.cpu().squeeze(0).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax[0].imshow(img)
            ax[0].set_title(f'Predicted: {class_names[preds.item()]}')

            prob_text = "\n".join([f"{class_names[j]}: {probs[0][j]:.4f}" for j in range(len(class_names))])
            ax[1].text(0.1, 0.5, prob_text, fontsize=12, verticalalignment='center')
            ax[1].axis('off')

            plt.tight_layout()
            plt.show()

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

visualize_random_predictions(model_best, dataloaders['test'], class_names)  #test ?? ?? ????


def load_image(image_data, transform=None):
# External data loading omitted for portfolio archive.
    if transform:
        image = transform(image).unsqueeze(0)   # ?
    return image

def predict_image(model, image_data, transform, class_names):
    image = load_image(image_data, transform)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    return preds.item(), probs.cpu().numpy()

def visualize_prediction(image_data, pred_class, pred_probs, class_names):
# External data loading omitted for portfolio archive.
    plt.imshow(image)
    plt.title(f'Predicted: {class_names[pred_class]}')
    plt.axis('off')
    plt.show()

    print("Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {pred_probs[0][i]:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import io
from IPython.display import display


n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)




import io
from IPython.display import display

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)




import io
from IPython.display import display

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)




import io
from IPython.display import display

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # ??fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)




