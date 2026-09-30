# Converted from the original Jupyter notebook.
# Markdown cells and saved text outputs are preserved as comments.

# %%
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

!pip install split-folders
import splitfolders

import cv2
import dlib
from torchvision.transforms import functional as F
# %% [markdown]
# Saved output
# Collecting split-folders
#   Downloading split_folders-0.5.1-py3-none-any.whl (8.4 kB)
# Installing collected packages: split-folders
# Successfully installed split-folders-0.5.1
# 
#
# %%
from google.colab import drive
drive.mount('/content/drive')
# %% [markdown]
# Saved output
# Mounted at /content/drive
# 
#
# %%
import zipfile

zip_path = '/content/drive/MyDrive/2024-1 ?묃뀬?먤뀯?メ꼱?듄넫?됣뀿?ⓤ????녲뀯?됣뀿?メ꼨?α꼥?듄녅/Team Project 3?뚡뀳/data.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/data')
# %%
# rootdirectory ?ㅼ젙
root_dir = "/content/data/6 Emotions for image classification"

# rootdirectory?덉뿉 ?덈뒗 ?곗씠?????class_names = sorted(x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))) # ?대옒???대쫫
num_class = len(class_names) # ?대옒??媛쒖닔
image_files = [ # 媛??대옒?ㅻ퀎 ?대?吏 寃쎈줈 由ъ뒪??    [os.path.join(root_dir, class_names[i], x) for x in os.listdir(os.path.join(root_dir, class_names[i]))]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)] # 媛??대옒?ㅻ퀎 ?대?吏 媛쒖닔
image_files_list = []
image_class = []

for i in range(num_class):
    image_files_list.extend(image_files[i]) # ?대?吏 寃쎈줈 由ъ뒪??    image_class.extend([i] * num_each[i]) # ?대?吏 ?쇰꺼 由ъ뒪??num_total = len(image_class)   # ?꾩껜 ?대?吏 媛쒖닔
image_width, image_height = PIL.Image.open(image_files_list[0]).size # ?대?吏 ?ъ씠利??뺤씤

data_dir = "data"
splitfolders.ratio(root_dir, data_dir, ratio=(.6,.2,.2),group_prefix=None) # train, val, test ?곗씠?곕줈 遺꾪븷

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")
# %% [markdown]
# Saved output
# Copying files: 1200 files [00:00, 4997.55 files/s]
# Total image count: 1200
# Image dimensions: 390 x 280
# Label names: ['anger', 'disgust', 'fear', 'happy', 'pain', 'sad']
# Label counts: [214, 201, 163, 230, 168, 224]
# 
# 
# 
#
# %%
# ?대?吏 異쒕젰
plt.subplots(3, 3, figsize=(8, 8)) # 3X3??洹몃┝
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <Figure size 800x800 with 9 Axes>
#
# %%
# ?쇨뎬 ?몄떇 ?⑥닔 ?뺤쓽
detector = dlib.get_frontal_face_detector()
predictor_path = "/content/drive/MyDrive/2024-1 ?묃뀬?먤뀯?メ꼱?듄넫?됣뀿?ⓤ????녲뀯?됣뀿?メ꼨?α꼥?듄녅/Team Project 3?뚡뀳/shape_predictor_68_face_landmarks.dat" # ?쇨뎬 ?대?吏?먯꽌 68媛쒖쓽 ?먯쑝濡??뱀젙 吏?먯쓣 寃異쒗븯湲??꾪븳 ?곗씠??predictor = dlib.shape_predictor(predictor_path)

def detect_and_crop_face(image):
    img = np.array(image) # ?대?吏瑜?numpy 諛곗뿴濡?蹂??
    # ?쇨뎬 ?먯?
    dets = detector(img, 1)
    if len(dets) > 0:
        d = dets[0] # 泥?踰덉㎏ ?쇨뎬 ?곸뿭 媛?몄삤湲?        left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
        # ?쇨뎬 ?곸뿭??寃쎄퀎瑜??대?吏 寃쎄퀎 ?덉뿉 ?덈뒗吏 ?뺤씤?섏뿬 ?덉쇅 泥섎━
        left = max(0, left)
        top = max(0, top)
        right = min(img.shape[1], right)
        bottom = min(img.shape[0], bottom)
        face = img[top:bottom, left:right] # ?쇨뎬 ?곸뿭???щ∼
        face_image = PIL.Image.fromarray(face) # ?쇨뎬 ?곸뿭??PIL ?대?吏濡?蹂??        return face_image
    else:
        return image   # ?쇨뎬??李얠? 紐삵븳 寃쎌슦 ?먮낯 ?대?吏瑜?諛섑솚
# %%
# transformation ?뺤쓽
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(detect_and_crop_face),  # ?쇨뎬 寃異?諛??щ∼
        transforms.Resize((224, 224)),  # ?ъ씠利?224x224濡??ъ젙??        transforms.RandomHorizontalFlip(),  # ?쒕뜡?쇰줈 醫뚯슦 諛섏쟾
        transforms.RandomRotation(30),  # ?쒕뜡?쇰줈 30???뚯쟾
        transforms.RandomAffine(degrees=0, shear=0.1, scale=(0.9, 1.1)),  # ?쒕뜡 ?댄뙆??蹂??        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # ?됱긽 蹂??        transforms.RandomGrayscale(p=0.1),  # 10% ?뺣쪧濡?洹몃젅?댁뒪耳??蹂??        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ?대?吏 ?뺢퇋??    ]),
    'val': transforms.Compose([
        transforms.Lambda(detect_and_crop_face),  # ?쇨뎬 寃異?諛??щ∼
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Lambda(detect_and_crop_face),  # ?쇨뎬 寃異?諛??щ∼
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ?곗씠?곗뀑 留뚮뱾湲?image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

# ?곗씠?곕줈??留뚮뱾湲?batch_size = 16
dataloaders = {x: torch.utils.data.DataLoader(
                    image_datasets[x],
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True) for x in ['train', 'val', 'test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']} # ?곗씠?곗뀑 ?ш린
class_names = image_datasets['train'].classes # ?대옒???대쫫

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU ?ъ슜 媛???щ????곕씪 device ?뺣낫 ???# %%
# ?꾩쿂由щ맂 ?대?吏 ?쒓컖??
inputs, classes = next(iter(dataloaders['train'])) # ?곗씠??濡쒕뜑?먯꽌 諛곗튂 媛?몄삤湲?
# ?덉떆 ?대?吏 異쒕젰
plt.subplots(3, 3, figsize=(8, 8))  # 3X3??洹몃┝
for i in range(9):  # 9媛쒖쓽 ?대?吏留?異쒕젰
    image = inputs[i]
    inp = image.permute(1, 2, 0).numpy() # ?대?吏 ?먯꽌瑜?numpy 諛곗뿴濡?蹂?섑븯???대?吏濡?蹂??    inp = np.clip(inp, 0, 1) # ?대?吏瑜?0怨?1 ?ъ씠??媛믪쑝濡??뺢퇋??
    plt.subplot(3, 3, i + 1)
    plt.imshow(inp)
    plt.title(class_names[classes[i]], fontsize=12)

plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <Figure size 800x800 with 9 Axes>
#
# %% [markdown]
# 
# 
# 
# ---
# 
# 
#
# %% [markdown]
# 
# **<肄붾뱶 ?쒖꽌 ?ㅻ챸>**
# 
# **1. 8媛쒖쓽 CNN model 鍮꾧탳**
# 
# **2. 4媛쒖쓽 Loss function 鍮꾧탳**
# 
# **3. 5媛쒖쓽 Optimizer 鍮꾧탳**
# 
# **4. test simulation**
# 
# **5. ?몃? ?대?吏 accuracy ?뺤씤**
# 
# **6. (異붽?) epoch=100 ?숈뒿**
#
# %%
# ?덈젴, 寃利? ?뚯뒪???곗씠?곗뀑 ?ш린 ?뺤씤
print(f"Train dataset size: {dataset_sizes['train']}")
print(f"Validation dataset size: {dataset_sizes['val']}")
print(f"Test dataset size: {dataset_sizes['test']}")
# %% [markdown]
# Saved output
# Train dataset size: 716
# Validation dataset size: 237
# Test dataset size: 245
# 
#
# %% [markdown]
# 
# ### **1. CNN model ?좎젙 - Loss function怨?Optimizer??怨좎젙(CELoss+Adam)**
#
# %% [markdown]
# 
# **[CNN Model]**
# 
# 1. Resnet50 : ILSVRC 2015 winner [?섏뾽?쒓컙???ъ슜]
# 
# - skip connection???댁슜??residual learning???듯빐 layer媛 源딆뼱吏먯뿉 ?곕Ⅸ vanishing gradient 臾몄젣瑜??닿껐?섎뒗 紐⑤뜽
# - 50 ?댁긽遺?곕뒗 parameter ?섎? ?쒖뼱?섍린 ?꾪빐 bottleneck 援ъ“??residual block???ъ슜??# 
# -------
# 
# 2. LeNet-5 : 珥덉갹湲?CNN model
# 
# - 1998??Yann LeCun???쇰Ц ?쁆radient-Based Learning Applied to Document
# Recognition?숈뿉 ?닿꺼 ?덈뒗 CNN ?좉꼍留앹쓽 援ъ“
# - ?쒗븳???뱀쭠留뚯쓣 異붿텧?섍퀬, ?낅젰媛믪쓽 topology瑜??ъ슜?섏? 紐삵븳?ㅻ뒗 ?⑥젏??媛吏?# - tanh ?쒖꽦???⑥닔 ?ъ슜
# 
# -------
# 
# 3. AlexNet : ILSVRC 2012 winner (top-5 err. 16.4%)
# 
# - GPU 2媛쒕? 蹂묐젹?곸쑝濡??ъ슜 媛??# - ReLU ?쒖꽦???⑥닔 ?ъ슜
# - Overlapping Pooling
# - Dropout 湲곕쾿 ?곸슜
# 
# -------
# 
# 4. VGGNet : ILSVRC 2014 2nd place (top-5 err. 7.3%)
# 
# - 紐⑤뱺 convoultion layer
# ?먯꽌 3*3 ?ш린??kernel???ъ슜?섏뿬 ?ㅽ듃?뚰겕??源딆씠媛 源딆뼱吏?
# 利? ?곸? ?뚮씪誘명꽣 ?섎줈 ???볦? receptive field瑜?媛뽰쓬
# 
# -------
# 
# 5. SENet : ILSVRC 2017 winner
# 
# - 3횞3 filter瑜?1횞1 filter濡??泥? ?낅젰 梨꾨꼸???섎? 媛먯냼?섏뿬 ?곗궛????땄
# 
# - SE block ?쒖븞
# 
# -------
# 
# 6. EfficientNet : 2018
# 
# - ?뚮씪誘명꽣 ?섎뒗 ?곴퀬, ?뺥솗?꾨뒗 利앷?
# 
# - 援ш? 釉뚮젅???? AutoML???듯빐 accuracy? flops??trade-off媛 理쒖쟻??媛源뚯슫 寃쎈웾 ?꾪궎?띿쿂 EfficientNet-B0 紐⑤뜽???쒖븞
# 
# -------
# 
# 7. DenseNet : 2017
# 
# - Resnet怨?鍮꾩듂?섍쾶 short connection???ъ슜?섏?留??욎쓽 ?덉씠?닿? 媛뽮퀬 ?덈뜕
# state瑜??댁뼱吏???덉씠?댁쓽 state??summation ?섎뒗 ???concatenating?섎뒗 諛⑹떇???ъ슜
# -bottleneck 援ъ“瑜??댁슜?섏뿬 ?뚮씪誘명꽣 ???듭젣
# 
# -------
# 
# 8. Resnet152(1踰??뺤옣)
# 
# - ResNet? 湲곗〈???쇰컲 ?ㅽ듃?뚰겕????щ━ 紐⑤뜽??源딆뼱吏덉닔濡??먮윭?⑥씠 ??븘吏???깅뒫??蹂댁엫
# - skip connection??layer媛 源딆뼱吏먯뿉 ?곕씪 諛쒖깮?섎뒗 vanishing gradient 臾몄젣瑜??닿껐?섍린?????ъ슫 理쒖쟻?붿? 源딆? ?ㅽ듃?뚰겕?먯꽌???뺥솗???μ긽??媛?ν븿
# 
# 
# 
# 
# 
# 
#
# %%
from torch.cuda.amp import GradScaler, autocast

# Mixed precision training ?ъ슜
scaler = GradScaler()

def train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs):
    file_name = f"{model_name}_model_params.pt"
    best_acc = 0.0
    loss_dict = {"train": [], "val": []}  #?먯떎 ????뺤뀛?덈━
    acc_dict = {"train": [], "val": []}   #?뺥솗??????뺤뀛?덈━

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   #?숈뒿 紐⑤뱶
            else:
                model.eval()    #?됯? 紐⑤뱶

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()     #?듯떚留덉씠? 湲곗슱湲?0?쇰줈 ?ㅼ젙

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():                   # Mixed precision training ?쒖꽦??                        outputs = model(inputs)
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
                scheduler.step(epoch_loss)   #?ㅼ?伊대윭 ?ъ슜

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    #異뷀썑 洹몃옒?꾨? 異쒕젰?섍린 ?꾪빐 ?먯떎/?뺥솗??湲곕줉
    torch.save(loss_dict, f"{model_name}_loss_dict.pth")
    torch.save(acc_dict, f"{model_name}_acc_dict.pth")
# %%
loss_function = torch.nn.CrossEntropyLoss() # loss function ?뺤쓽
max_epochs = 30 # epoch ???뺤쓽
# %% [markdown]
# 
# **[Mixed precision training]**
# 
# https://pytorch.org/docs/stable/amp.html 李멸퀬??# 
# 硫붾え由??ъ슜??媛먯냼 + ?곗궛 ?띾룄 ?μ긽???꾪빐 異붽?
#
# %% [markdown]
# 
# **[scheduler ?ъ슜]**
# 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# 
# : ?먯떎?????댁긽 媛먯냼?섏? ?딆쓣???숈뒿瑜좎쓣 ?좊룞?곸쑝濡?以꾩엫
# 
# -> epoch=10?숈븞 吏?쒓? 媛쒖꽑?섏? ?딆쑝硫??숈뒿瑜?媛먯냼(10%)?쒗궎?꾨줉 ?ㅼ젙
# 
#
# %% [markdown]
# 
# ####**1. resnet50 (?섏뾽?쒓컙怨??숈씪)**
#
# %%
n_features = 6  #6媛쒖쓽 ?대옒??
model_1 = models.resnet50(pretrained=True)  # resnet50 紐⑤뜽 遺덈윭?ㅺ린   # pretrained: ?좏븰?듬맂 ?곗씠?곕? 媛吏怨??숈뒿?섍쿋??
model_1.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout ?ъ슜 -> overfitting 諛⑹?
    nn.Linear(model_1.fc.in_features, n_features)  # 留덉?留?fully connected layer
)

model_1.to(device)   # 紐⑤뜽???붾컮?댁뒪濡?蹂대궡以섏빞 ??(CPU -> GPU)

model_1  # 紐⑤뜽 異쒕젰
# %% [markdown]
# Saved output
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=2048, out_features=6, bias=True)
#   )
# )
#
# %%
optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_1, 'resnet50', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.7776 Acc: 0.2221
# val Total Loss: 1.7005 Acc: 0.3460
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6713 Acc: 0.3226
# val Total Loss: 1.6095 Acc: 0.4135
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5593 Acc: 0.3813
# val Total Loss: 1.4974 Acc: 0.4810
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.4300 Acc: 0.4874
# val Total Loss: 1.3799 Acc: 0.5359
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.2805 Acc: 0.5419
# val Total Loss: 1.2764 Acc: 0.5527
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.2130 Acc: 0.5768
# val Total Loss: 1.1950 Acc: 0.5865
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.0887 Acc: 0.6369
# val Total Loss: 1.1365 Acc: 0.5823
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.0220 Acc: 0.6578
# val Total Loss: 1.0693 Acc: 0.6203
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.9426 Acc: 0.6927
# val Total Loss: 1.0218 Acc: 0.6329
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.8429 Acc: 0.7388
# val Total Loss: 0.9874 Acc: 0.6498
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.8194 Acc: 0.7179
# val Total Loss: 0.9582 Acc: 0.6751
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.7422 Acc: 0.7709
# val Total Loss: 0.9455 Acc: 0.6793
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.6881 Acc: 0.7696
# val Total Loss: 0.9156 Acc: 0.7004
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.6615 Acc: 0.7933
# val Total Loss: 0.9378 Acc: 0.6878
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.6017 Acc: 0.8268
# val Total Loss: 0.8971 Acc: 0.7004
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.5705 Acc: 0.8268
# val Total Loss: 0.8912 Acc: 0.7004
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.5130 Acc: 0.8422
# val Total Loss: 0.8906 Acc: 0.6962
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.4931 Acc: 0.8436
# val Total Loss: 0.8539 Acc: 0.7131
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.4046 Acc: 0.8966
# val Total Loss: 0.8715 Acc: 0.7046
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.4063 Acc: 0.8757
# val Total Loss: 0.8719 Acc: 0.7046
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.3867 Acc: 0.8827
# val Total Loss: 0.8731 Acc: 0.7131
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.3712 Acc: 0.8911
# val Total Loss: 0.8659 Acc: 0.7046
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.3254 Acc: 0.9078
# val Total Loss: 0.8556 Acc: 0.7004
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.3005 Acc: 0.9246
# val Total Loss: 0.8901 Acc: 0.7004
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.2769 Acc: 0.9372
# val Total Loss: 0.8643 Acc: 0.7173
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.2530 Acc: 0.9372
# val Total Loss: 0.8862 Acc: 0.7046
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.2289 Acc: 0.9441
# val Total Loss: 0.8830 Acc: 0.7089
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.2655 Acc: 0.9330
# val Total Loss: 0.8668 Acc: 0.7131
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.2126 Acc: 0.9553
# val Total Loss: 0.8780 Acc: 0.7173
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.2190 Acc: 0.9399
# val Total Loss: 0.8643 Acc: 0.7131
# 
# Best val Acc: 0.7173
# 
#
# %% [markdown]
# 
# ####**2. LeNet-5**
# 
# https://stydy-sturdy.tistory.com/5 李멸퀬?섏??듬땲??
#
# %%
n_features = 6

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5, 1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5, 1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 6),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# %%
model_2 = LeNet5()
model_2.to(device)  # 紐⑤뜽??GPU ?먮뒗 CPU濡??대룞
model_2
# %% [markdown]
# Saved output
# LeNet5(
#   (feature_extractor): Sequential(
#     (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
#     (1): Tanh()
#     (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#     (4): Tanh()
#     (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     (6): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
#     (7): Tanh()
#     (8): AdaptiveAvgPool2d(output_size=(1, 1))
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=120, out_features=84, bias=True)
#     (1): Tanh()
#     (2): Linear(in_features=84, out_features=6, bias=True)
#   )
# )
#
# %%
optimizer = torch.optim.Adam(model_2.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# 
#
# %%
train_model(model_2, 'LeNet5', loss_function, optimizer, scheduler , max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.7863 Acc: 0.1913
# val Total Loss: 1.7894 Acc: 0.1899
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.7808 Acc: 0.1927
# val Total Loss: 1.7860 Acc: 0.1899
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.7771 Acc: 0.1927
# val Total Loss: 1.7833 Acc: 0.1941
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.7731 Acc: 0.1969
# val Total Loss: 1.7800 Acc: 0.1941
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.7705 Acc: 0.2011
# val Total Loss: 1.7768 Acc: 0.2025
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.7656 Acc: 0.2151
# val Total Loss: 1.7750 Acc: 0.1941
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.7606 Acc: 0.2151
# val Total Loss: 1.7724 Acc: 0.2025
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.7568 Acc: 0.2235
# val Total Loss: 1.7703 Acc: 0.2025
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.7554 Acc: 0.2207
# val Total Loss: 1.7687 Acc: 0.2025
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.7507 Acc: 0.2249
# val Total Loss: 1.7664 Acc: 0.2110
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.7452 Acc: 0.2277
# val Total Loss: 1.7648 Acc: 0.2152
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.7430 Acc: 0.2263
# val Total Loss: 1.7637 Acc: 0.2110
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.7408 Acc: 0.2291
# val Total Loss: 1.7629 Acc: 0.2194
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.7413 Acc: 0.2235
# val Total Loss: 1.7619 Acc: 0.2152
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.7368 Acc: 0.2444
# val Total Loss: 1.7609 Acc: 0.2236
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.7348 Acc: 0.2360
# val Total Loss: 1.7599 Acc: 0.2278
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.7392 Acc: 0.2416
# val Total Loss: 1.7595 Acc: 0.2447
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.7375 Acc: 0.2374
# val Total Loss: 1.7587 Acc: 0.2489
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.7258 Acc: 0.2486
# val Total Loss: 1.7580 Acc: 0.2532
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.7348 Acc: 0.2346
# val Total Loss: 1.7573 Acc: 0.2574
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.7330 Acc: 0.2249
# val Total Loss: 1.7564 Acc: 0.2616
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.7275 Acc: 0.2402
# val Total Loss: 1.7552 Acc: 0.2616
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.7284 Acc: 0.2444
# val Total Loss: 1.7538 Acc: 0.2658
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.7260 Acc: 0.2430
# val Total Loss: 1.7532 Acc: 0.2743
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.7248 Acc: 0.2318
# val Total Loss: 1.7526 Acc: 0.2743
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.7218 Acc: 0.2500
# val Total Loss: 1.7519 Acc: 0.2743
# 
# Epoch 27/30
# ----------
# train Total Loss: 1.7194 Acc: 0.2458
# val Total Loss: 1.7514 Acc: 0.2658
# 
# Epoch 28/30
# ----------
# train Total Loss: 1.7241 Acc: 0.2472
# val Total Loss: 1.7508 Acc: 0.2658
# 
# Epoch 29/30
# ----------
# train Total Loss: 1.7156 Acc: 0.2598
# val Total Loss: 1.7503 Acc: 0.2700
# 
# Epoch 30/30
# ----------
# train Total Loss: 1.7207 Acc: 0.2528
# val Total Loss: 1.7501 Acc: 0.2827
# 
# Best val Acc: 0.2827
# 
#
# %% [markdown]
# 
# ####**3. AlexNet**
#
# %%
n_features = 6

model_3 = models.alexnet(pretrained=True)
model_3.classifier[6] = nn.Linear(model_3.classifier[6].in_features, n_features)
model_3.to(device)   # 紐⑤뜽???붾컮?댁뒪濡?蹂대궡以섏빞 ??(CPU -> GPU)

optimizer = torch.optim.Adam(model_3.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_3, 'AlexNet', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# 100%|?댿뻽?댿뻽?댿뻽?댿뻽?댿뻽| 233M/233M [00:01<00:00, 184MB/s]
# 
# Epoch 1/30
# ----------
# 
# /usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
#   return F.conv2d(input, weight, bias, self.stride,
# 
# train Total Loss: 1.8471 Acc: 0.1774
# val Total Loss: 1.6877 Acc: 0.2827
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6391 Acc: 0.3184
# val Total Loss: 1.5758 Acc: 0.3671
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5088 Acc: 0.4022
# val Total Loss: 1.4813 Acc: 0.4135
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.4292 Acc: 0.4232
# val Total Loss: 1.3969 Acc: 0.4473
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.3403 Acc: 0.4791
# val Total Loss: 1.3276 Acc: 0.4684
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.2350 Acc: 0.5209
# val Total Loss: 1.2885 Acc: 0.4852
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.1569 Acc: 0.5642
# val Total Loss: 1.2208 Acc: 0.5359
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.0686 Acc: 0.5978
# val Total Loss: 1.1798 Acc: 0.5274
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.0250 Acc: 0.6411
# val Total Loss: 1.1495 Acc: 0.5527
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.0037 Acc: 0.6159
# val Total Loss: 1.1238 Acc: 0.5485
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.9414 Acc: 0.6355
# val Total Loss: 1.0921 Acc: 0.5527
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.8936 Acc: 0.6411
# val Total Loss: 1.1171 Acc: 0.5738
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.8394 Acc: 0.6732
# val Total Loss: 1.1204 Acc: 0.5865
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.8244 Acc: 0.7025
# val Total Loss: 1.0880 Acc: 0.5865
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.7588 Acc: 0.7109
# val Total Loss: 1.0745 Acc: 0.6118
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.7412 Acc: 0.7235
# val Total Loss: 1.1020 Acc: 0.6329
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.6875 Acc: 0.7458
# val Total Loss: 1.0664 Acc: 0.6414
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.6535 Acc: 0.7612
# val Total Loss: 1.0408 Acc: 0.6582
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.6477 Acc: 0.7668
# val Total Loss: 1.0667 Acc: 0.6329
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.6490 Acc: 0.7612
# val Total Loss: 1.0579 Acc: 0.6329
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.6189 Acc: 0.7793
# val Total Loss: 1.0678 Acc: 0.6456
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.5767 Acc: 0.7933
# val Total Loss: 1.0546 Acc: 0.6371
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.5455 Acc: 0.8045
# val Total Loss: 1.0562 Acc: 0.6540
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.5086 Acc: 0.8087
# val Total Loss: 1.0166 Acc: 0.6624
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.4903 Acc: 0.8212
# val Total Loss: 1.0615 Acc: 0.6498
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.4350 Acc: 0.8422
# val Total Loss: 1.0965 Acc: 0.6498
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.4718 Acc: 0.8366
# val Total Loss: 1.0560 Acc: 0.6498
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.3979 Acc: 0.8659
# val Total Loss: 1.0981 Acc: 0.6245
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.4203 Acc: 0.8520
# val Total Loss: 1.0817 Acc: 0.6371
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.3762 Acc: 0.8715
# val Total Loss: 1.0959 Acc: 0.6498
# 
# Best val Acc: 0.6624
# 
#
# %% [markdown]
# 
# ####**4. Vggnet**
#
# %%
n_features = 6
model_4 = models.vgg16(pretrained=True)
num_features = model_4.classifier[6].in_features
model_4.classifier[6] = nn.Linear(num_features, n_features)
model_4.to(device)
optimizer = torch.optim.Adam(model_4.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
train_model(model_4, 'Vggnet', loss_function, optimizer,scheduler,max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.8211 Acc: 0.1816
# val Total Loss: 1.7467 Acc: 0.2447
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6577 Acc: 0.2933
# val Total Loss: 1.6587 Acc: 0.3207
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5209 Acc: 0.3729
# val Total Loss: 1.4975 Acc: 0.3966
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.3771 Acc: 0.4469
# val Total Loss: 1.4127 Acc: 0.4262
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.2438 Acc: 0.4888
# val Total Loss: 1.2648 Acc: 0.5570
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.1461 Acc: 0.5698
# val Total Loss: 1.1398 Acc: 0.6160
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.9809 Acc: 0.6229
# val Total Loss: 1.1187 Acc: 0.6118
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.9387 Acc: 0.6411
# val Total Loss: 1.0430 Acc: 0.6414
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.8138 Acc: 0.7025
# val Total Loss: 1.0643 Acc: 0.6414
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.7665 Acc: 0.7095
# val Total Loss: 1.0180 Acc: 0.6667
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.7326 Acc: 0.7221
# val Total Loss: 1.0235 Acc: 0.6287
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.6759 Acc: 0.7430
# val Total Loss: 0.9840 Acc: 0.6540
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.5683 Acc: 0.8017
# val Total Loss: 1.0046 Acc: 0.6751
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.5355 Acc: 0.8156
# val Total Loss: 0.9499 Acc: 0.6835
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.5259 Acc: 0.8282
# val Total Loss: 0.9757 Acc: 0.6878
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.4826 Acc: 0.8212
# val Total Loss: 0.9499 Acc: 0.7089
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.4268 Acc: 0.8436
# val Total Loss: 1.0001 Acc: 0.6878
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.4118 Acc: 0.8561
# val Total Loss: 0.9681 Acc: 0.7046
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.3544 Acc: 0.8785
# val Total Loss: 1.0881 Acc: 0.6540
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.3687 Acc: 0.8617
# val Total Loss: 1.0131 Acc: 0.6835
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.3062 Acc: 0.8966
# val Total Loss: 1.0324 Acc: 0.6835
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.2477 Acc: 0.9148
# val Total Loss: 1.0580 Acc: 0.6920
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.2638 Acc: 0.9078
# val Total Loss: 1.0917 Acc: 0.7131
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.2350 Acc: 0.9134
# val Total Loss: 1.1040 Acc: 0.7004
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.2189 Acc: 0.9232
# val Total Loss: 1.2115 Acc: 0.6962
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.2051 Acc: 0.9330
# val Total Loss: 1.1464 Acc: 0.7089
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.1650 Acc: 0.9497
# val Total Loss: 1.1619 Acc: 0.6962
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.1425 Acc: 0.9567
# val Total Loss: 1.1638 Acc: 0.6920
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.1654 Acc: 0.9399
# val Total Loss: 1.1758 Acc: 0.6962
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.1368 Acc: 0.9511
# val Total Loss: 1.1787 Acc: 0.7046
# 
# Best val Acc: 0.7131
# 
#
# %% [markdown]
# 
# ####**5. SEnet**
#
# %%
n_features = 6
model_5 = models.squeezenet1_0(pretrained=True)
model_5.classifier[1] = nn.Conv2d(512, n_features, kernel_size=1)
model_5.to(device)
optimizer = torch.optim.Adam(model_5.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
train_model(model_5, 'SEnet', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=SqueezeNet1_0_Weights.IMAGENET1K_V1`. You can also use `weights=SqueezeNet1_0_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth" to /root/.cache/torch/hub/checkpoints/squeezenet1_0-b66bff10.pth
# 100%|?댿뻽?댿뻽?댿뻽?댿뻽?댿뻽| 4.78M/4.78M [00:00<00:00, 119MB/s]
# Epoch 1/30
# ----------
# 
# 
# 
# train Total Loss: 1.8894 Acc: 0.2165
# val Total Loss: 1.8262 Acc: 0.2616
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.7244 Acc: 0.2849
# val Total Loss: 1.7632 Acc: 0.2616
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.6838 Acc: 0.2835
# val Total Loss: 1.7266 Acc: 0.2785
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.6398 Acc: 0.3366
# val Total Loss: 1.6594 Acc: 0.2827
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.5820 Acc: 0.3464
# val Total Loss: 1.6181 Acc: 0.3249
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.5287 Acc: 0.3953
# val Total Loss: 1.5440 Acc: 0.3629
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.4782 Acc: 0.4288
# val Total Loss: 1.4776 Acc: 0.3797
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.4252 Acc: 0.4316
# val Total Loss: 1.4569 Acc: 0.4473
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.3852 Acc: 0.4693
# val Total Loss: 1.4595 Acc: 0.4430
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.3427 Acc: 0.4846
# val Total Loss: 1.4041 Acc: 0.4599
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.3126 Acc: 0.4930
# val Total Loss: 1.3778 Acc: 0.4768
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.2975 Acc: 0.5084
# val Total Loss: 1.3508 Acc: 0.4852
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.2928 Acc: 0.5056
# val Total Loss: 1.3237 Acc: 0.5063
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.2312 Acc: 0.5237
# val Total Loss: 1.3138 Acc: 0.5232
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.2384 Acc: 0.5349
# val Total Loss: 1.2855 Acc: 0.5232
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.2112 Acc: 0.5503
# val Total Loss: 1.2708 Acc: 0.5443
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.1613 Acc: 0.5712
# val Total Loss: 1.2887 Acc: 0.5485
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.1761 Acc: 0.5642
# val Total Loss: 1.2841 Acc: 0.5570
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.1678 Acc: 0.5656
# val Total Loss: 1.2683 Acc: 0.5485
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.1350 Acc: 0.5740
# val Total Loss: 1.2374 Acc: 0.5612
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.1226 Acc: 0.5587
# val Total Loss: 1.2435 Acc: 0.5401
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.1099 Acc: 0.5726
# val Total Loss: 1.2397 Acc: 0.5823
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.0848 Acc: 0.6034
# val Total Loss: 1.2547 Acc: 0.5570
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.1015 Acc: 0.5922
# val Total Loss: 1.2940 Acc: 0.5316
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.0929 Acc: 0.5824
# val Total Loss: 1.2319 Acc: 0.5612
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.0254 Acc: 0.6215
# val Total Loss: 1.3338 Acc: 0.5190
# 
# Epoch 27/30
# ----------
# train Total Loss: 1.0281 Acc: 0.6271
# val Total Loss: 1.2457 Acc: 0.5654
# 
# Epoch 28/30
# ----------
# train Total Loss: 1.0012 Acc: 0.6369
# val Total Loss: 1.2595 Acc: 0.5527
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.9998 Acc: 0.6313
# val Total Loss: 1.1868 Acc: 0.6034
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.9947 Acc: 0.6453
# val Total Loss: 1.2092 Acc: 0.5949
# 
# Best val Acc: 0.6034
# 
#
# %% [markdown]
# 
# ####**6. effiecientNet**
#
# %%
!pip install timm
import timm
# %% [markdown]
# Saved output
# Collecting timm
#   Downloading timm-1.0.3-py3-none-any.whl (2.3 MB)
# [?25l     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m0.0/2.3 MB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺??[0m[90m??[0m[90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m0.8/2.3 MB[0m [31m22.8 MB/s[0m eta [36m0:00:01[0m[2K     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m2.3/2.3 MB[0m [31m38.2 MB/s[0m eta [36m0:00:00[0m
# [?25hRequirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from timm) (2.3.0+cu121)
# Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from timm) (0.18.0+cu121)
# Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from timm) (6.0.1)
# Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.10/dist-packages (from timm) (0.23.1)
# Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from timm) (0.4.3)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (3.14.0)
# Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2023.6.0)
# Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (24.0)
# Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (2.31.0)
# Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.66.4)
# Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface_hub->timm) (4.11.0)
# Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->timm) (1.12)
# Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.3)
# Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (3.1.4)
# Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch->timm)
#   Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
# Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch->timm)
#   Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
# Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch->timm)
#   Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
# Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch->timm)
#   Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
# Collecting nvidia-cublas-cu12==12.1.3.1 (from torch->timm)
#   Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
# Collecting nvidia-cufft-cu12==11.0.2.54 (from torch->timm)
#   Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
# Collecting nvidia-curand-cu12==10.3.2.106 (from torch->timm)
#   Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
# Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch->timm)
#   Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
# Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch->timm)
#   Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
# Collecting nvidia-nccl-cu12==2.20.5 (from torch->timm)
#   Using cached nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
# Collecting nvidia-nvtx-cu12==12.1.105 (from torch->timm)
#   Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
# Requirement already satisfied: triton==2.3.0 in /usr/local/lib/python3.10/dist-packages (from torch->timm) (2.3.0)
# Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch->timm)
#   Downloading nvidia_nvjitlink_cu12-12.5.40-py3-none-manylinux2014_x86_64.whl (21.3 MB)
# [2K     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m21.3/21.3 MB[0m [31m71.7 MB/s[0m eta [36m0:00:00[0m
# [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (1.25.2)
# Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (9.4.0)
# Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->timm) (2.1.5)
# Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.3.2)
# Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (3.7)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2.0.7)
# Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface_hub->timm) (2024.2.2)
# Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->timm) (1.3.0)
# Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, timm
# Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.5.40 nvidia-nvtx-cu12-12.1.105 timm-1.0.3
# 
#
# %%
n_features = 6

model_6 = timm.create_model("efficientnet_b4", pretrained=True) #pretrained : ?좏븰?듬맂 ?곗씠?곕? 媛吏怨??숈뒿?섍쿋??
model_6.classifier = nn.Linear(model_6.classifier.in_features, n_features)
model_6.to(device)

optimizer = torch.optim.Adam(model_6.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_6, 'EfficientNet', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
# The secret `HF_TOKEN` does not exist in your Colab secrets.
# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
# You will be able to reuse this secret in all of your notebooks.
# Please note that authentication is recommended but still optional to access public models or datasets.
#   warnings.warn(
# 
# model.safetensors:   0%|          | 0.00/77.9M [00:00<?, ?B/s]
# Epoch 1/30
# ----------
# train Total Loss: 1.7975 Acc: 0.1592
# val Total Loss: 1.7982 Acc: 0.1730
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.7867 Acc: 0.1941
# val Total Loss: 1.7934 Acc: 0.1730
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.7823 Acc: 0.2123
# val Total Loss: 1.7875 Acc: 0.1814
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.7683 Acc: 0.2500
# val Total Loss: 1.7823 Acc: 0.1772
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.7632 Acc: 0.2709
# val Total Loss: 1.7780 Acc: 0.2236
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.7575 Acc: 0.2849
# val Total Loss: 1.7708 Acc: 0.2489
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.7489 Acc: 0.3101
# val Total Loss: 1.7659 Acc: 0.2658
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.7448 Acc: 0.3003
# val Total Loss: 1.7581 Acc: 0.2911
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.7324 Acc: 0.3506
# val Total Loss: 1.7534 Acc: 0.2911
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.7195 Acc: 0.3715
# val Total Loss: 1.7472 Acc: 0.3249
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.7116 Acc: 0.3757
# val Total Loss: 1.7381 Acc: 0.3291
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.7025 Acc: 0.3673
# val Total Loss: 1.7276 Acc: 0.3122
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.6894 Acc: 0.4008
# val Total Loss: 1.7188 Acc: 0.3755
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.6748 Acc: 0.4134
# val Total Loss: 1.7085 Acc: 0.3586
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.6661 Acc: 0.4022
# val Total Loss: 1.6987 Acc: 0.3629
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.6512 Acc: 0.3980
# val Total Loss: 1.6876 Acc: 0.3544
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.6381 Acc: 0.4372
# val Total Loss: 1.6740 Acc: 0.3882
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.6173 Acc: 0.4358
# val Total Loss: 1.6563 Acc: 0.3924
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.5977 Acc: 0.4455
# val Total Loss: 1.6464 Acc: 0.3840
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.5820 Acc: 0.4427
# val Total Loss: 1.6284 Acc: 0.3966
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.5591 Acc: 0.4679
# val Total Loss: 1.6116 Acc: 0.4135
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.5393 Acc: 0.4651
# val Total Loss: 1.5947 Acc: 0.4177
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.5157 Acc: 0.4986
# val Total Loss: 1.5708 Acc: 0.4430
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.4960 Acc: 0.4888
# val Total Loss: 1.5501 Acc: 0.4557
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.4690 Acc: 0.5279
# val Total Loss: 1.5307 Acc: 0.4557
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.4466 Acc: 0.5391
# val Total Loss: 1.5037 Acc: 0.4684
# 
# Epoch 27/30
# ----------
# train Total Loss: 1.4245 Acc: 0.5307
# val Total Loss: 1.4879 Acc: 0.4852
# 
# Epoch 28/30
# ----------
# train Total Loss: 1.3897 Acc: 0.5712
# val Total Loss: 1.4607 Acc: 0.4979
# 
# Epoch 29/30
# ----------
# train Total Loss: 1.3640 Acc: 0.5740
# val Total Loss: 1.4325 Acc: 0.5105
# 
# Epoch 30/30
# ----------
# train Total Loss: 1.3456 Acc: 0.5796
# val Total Loss: 1.4079 Acc: 0.5232
# 
# Best val Acc: 0.5232
# 
#
# %% [markdown]
# 
# ####**7. Densenet**
#
# %%
import timm

n_features = 6

# DenseNet 紐⑤뜽 ?뺤쓽
model_7 = timm.create_model("densenet121", pretrained=True) # pretrained: ?좏븰?듬맂 ?곗씠?곕? ?ъ슜
model_7.classifier = nn.Linear(model_7.classifier.in_features, n_features) # 異쒕젰 ?ш린瑜?6?쇰줈 蹂寃?model_7.to(device)

optimizer = torch.optim.Adam(model_7.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_7, 'Densenet', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# model.safetensors:   0%|          | 0.00/32.3M [00:00<?, ?B/s]
# Epoch 1/30
# ----------
# train Total Loss: 1.8343 Acc: 0.1732
# val Total Loss: 1.8369 Acc: 0.1814
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.7673 Acc: 0.2277
# val Total Loss: 1.8004 Acc: 0.2194
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.7239 Acc: 0.2668
# val Total Loss: 1.7714 Acc: 0.2489
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.6839 Acc: 0.3170
# val Total Loss: 1.7350 Acc: 0.2616
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.6559 Acc: 0.3254
# val Total Loss: 1.7110 Acc: 0.2996
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.6280 Acc: 0.3715
# val Total Loss: 1.6833 Acc: 0.3080
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.5878 Acc: 0.3980
# val Total Loss: 1.6536 Acc: 0.3291
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.5429 Acc: 0.4413
# val Total Loss: 1.6255 Acc: 0.3671
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.4942 Acc: 0.4902
# val Total Loss: 1.6002 Acc: 0.3671
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.4616 Acc: 0.4958
# val Total Loss: 1.5650 Acc: 0.3713
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.4419 Acc: 0.5042
# val Total Loss: 1.5403 Acc: 0.3755
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.3897 Acc: 0.5196
# val Total Loss: 1.5164 Acc: 0.4177
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.3480 Acc: 0.5642
# val Total Loss: 1.4773 Acc: 0.4304
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.3171 Acc: 0.5656
# val Total Loss: 1.4426 Acc: 0.4473
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.2928 Acc: 0.5754
# val Total Loss: 1.4180 Acc: 0.4599
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.2686 Acc: 0.6034
# val Total Loss: 1.3806 Acc: 0.5021
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.2313 Acc: 0.6047
# val Total Loss: 1.3550 Acc: 0.5063
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.1977 Acc: 0.6173
# val Total Loss: 1.3340 Acc: 0.5148
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.1818 Acc: 0.6439
# val Total Loss: 1.3032 Acc: 0.5612
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.1271 Acc: 0.6425
# val Total Loss: 1.2714 Acc: 0.5612
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.1033 Acc: 0.6648
# val Total Loss: 1.2544 Acc: 0.5654
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.0905 Acc: 0.6466
# val Total Loss: 1.2288 Acc: 0.5654
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.0537 Acc: 0.6564
# val Total Loss: 1.2012 Acc: 0.5865
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.0275 Acc: 0.7039
# val Total Loss: 1.1856 Acc: 0.5865
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.9947 Acc: 0.6802
# val Total Loss: 1.1696 Acc: 0.6076
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.0057 Acc: 0.6899
# val Total Loss: 1.1522 Acc: 0.5992
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.9646 Acc: 0.7025
# val Total Loss: 1.1229 Acc: 0.6203
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.9306 Acc: 0.7067
# val Total Loss: 1.1088 Acc: 0.6329
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.9258 Acc: 0.7291
# val Total Loss: 1.1136 Acc: 0.6203
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.9203 Acc: 0.7235
# val Total Loss: 1.0838 Acc: 0.6371
# 
# Best val Acc: 0.6371
# 
#
# %% [markdown]
# 
# ####**8.resnet152**
#
# %%
n_features = 6

model_8 = models.resnet152(pretrained=True)
model_8.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_8.fc.in_features, n_features)
)
model_8.to(device)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/resnet152-394f9c45.pth" to /root/.cache/torch/hub/checkpoints/resnet152-394f9c45.pth
# 100%|?댿뻽?댿뻽?댿뻽?댿뻽?댿뻽| 230M/230M [00:01<00:00, 237MB/s]
# 
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (6): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (7): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (6): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (7): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (8): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (9): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (10): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (11): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (12): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (13): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (14): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (15): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (16): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (17): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (18): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (19): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (20): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (21): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (22): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (23): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (24): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (25): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (26): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (27): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (28): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (29): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (30): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (31): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (32): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (33): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (34): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (35): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=2048, out_features=6, bias=True)
#   )
# )
#
# %%
optimizer = torch.optim.Adam(model_8.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
# %%
train_model(model_8, 'resnet152', loss_function, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.8116 Acc: 0.1760
# val Total Loss: 1.7004 Acc: 0.2658
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6451 Acc: 0.3073
# val Total Loss: 1.5806 Acc: 0.3755
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5197 Acc: 0.4204
# val Total Loss: 1.4662 Acc: 0.4768
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.4034 Acc: 0.4777
# val Total Loss: 1.3223 Acc: 0.5612
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.2547 Acc: 0.5587
# val Total Loss: 1.2272 Acc: 0.5738
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.1408 Acc: 0.5964
# val Total Loss: 1.0947 Acc: 0.6245
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.9947 Acc: 0.6634
# val Total Loss: 1.0161 Acc: 0.6287
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.9003 Acc: 0.6983
# val Total Loss: 0.9465 Acc: 0.6709
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.8060 Acc: 0.7207
# val Total Loss: 0.9103 Acc: 0.6793
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.7302 Acc: 0.7696
# val Total Loss: 0.8801 Acc: 0.6667
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.6696 Acc: 0.7961
# val Total Loss: 0.8824 Acc: 0.6962
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.5960 Acc: 0.8115
# val Total Loss: 0.8354 Acc: 0.7046
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.5519 Acc: 0.8184
# val Total Loss: 0.8067 Acc: 0.7342
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.4628 Acc: 0.8687
# val Total Loss: 0.8099 Acc: 0.7300
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.4558 Acc: 0.8645
# val Total Loss: 0.8040 Acc: 0.7300
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.3838 Acc: 0.8841
# val Total Loss: 0.7637 Acc: 0.7342
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.3491 Acc: 0.8953
# val Total Loss: 0.7740 Acc: 0.7257
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.2777 Acc: 0.9288
# val Total Loss: 0.8060 Acc: 0.7215
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.2788 Acc: 0.9246
# val Total Loss: 0.7985 Acc: 0.7426
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.2408 Acc: 0.9316
# val Total Loss: 0.8326 Acc: 0.7215
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.2692 Acc: 0.9218
# val Total Loss: 0.8041 Acc: 0.7342
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.2286 Acc: 0.9413
# val Total Loss: 0.8200 Acc: 0.7342
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.1766 Acc: 0.9539
# val Total Loss: 0.8228 Acc: 0.7384
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.1714 Acc: 0.9553
# val Total Loss: 0.8311 Acc: 0.7173
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.1641 Acc: 0.9651
# val Total Loss: 0.8418 Acc: 0.7215
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.1533 Acc: 0.9665
# val Total Loss: 0.8606 Acc: 0.7215
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.1574 Acc: 0.9511
# val Total Loss: 0.9108 Acc: 0.7131
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.1291 Acc: 0.9651
# val Total Loss: 0.8517 Acc: 0.7257
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.1131 Acc: 0.9791
# val Total Loss: 0.8473 Acc: 0.7426
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.1220 Acc: 0.9637
# val Total Loss: 0.8383 Acc: 0.7257
# 
# Best val Acc: 0.7426
# 
#
# %% [markdown]
# 
# ####**model_1~8 Evaluation**
#
# %%
plt.figure(figsize=(12, 8))

resnet50_acc_dict = torch.load("/content/resnet50_acc_dict.pth")
plt.plot(range(1, len(resnet50_acc_dict['val']) + 1), resnet50_acc_dict['val'], label='resnet50')

LeNet5_acc_dict = torch.load("/content/LeNet5_acc_dict.pth")
plt.plot(range(1, len(LeNet5_acc_dict['val']) + 1), LeNet5_acc_dict['val'], label='LeNet5')

AlexNet_acc_dict = torch.load("/content/AlexNet_acc_dict.pth")
plt.plot(range(1, len(AlexNet_acc_dict['val']) + 1), AlexNet_acc_dict['val'], label='AlexNet')

Vggnet_acc_dict = torch.load("/content/Vggnet_acc_dict.pth")
plt.plot(range(1, len(Vggnet_acc_dict['val']) + 1), Vggnet_acc_dict['val'], label='Vggnet')

SEnet_acc_dict = torch.load("/content/SEnet_acc_dict.pth")
plt.plot(range(1, len(SEnet_acc_dict['val']) + 1), SEnet_acc_dict['val'], label='SEnet')

EfficientNet_acc_dict = torch.load("/content/EfficientNet_acc_dict.pth")
plt.plot(range(1, len(EfficientNet_acc_dict['val']) + 1), EfficientNet_acc_dict['val'], label='EfficientNet')

DenseNet_acc_dict = torch.load("/content/Densenet_acc_dict.pth")
plt.plot(range(1, len(DenseNet_acc_dict['val']) + 1), DenseNet_acc_dict['val'], label='DenseNet')

resnet152_acc_dict = torch.load("/content/resnet152_acc_dict.pth")
plt.plot(range(1, len(resnet152_acc_dict['val']) + 1), resnet152_acc_dict['val'], label='resnet152')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy of Models')
plt.savefig('validation_accuracy.png')
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1200x800 with 1 Axes>
#
# %%
model_best=model_8
# %% [markdown]
# 
# ### **2. Loss function ?좎젙 - Optimizer??怨좎젙(Adam)**
#
# %% [markdown]
# 
# **[Loss function]**
# 
# CrossEntropyLoss : ?ㅼ쨷 ?대옒??遺꾨쪟 / LogSoftmax ?ы븿
# 
# MSELoss : ?뚭?紐⑤뜽 ?ъ슜
# 
# MultiLabelSoftMarginLoss : ?ㅼ쨷 ?대옒??遺꾨쪟, 媛??섑뵆???щ윭 ?덉씠釉?媛吏????덉쓬
# 
# MultiMarginLoss : ?ㅼ쨷 ?대옒??遺꾨쪟 , 媛??대옒?ㅼ뿉 ???留덉쭊 理쒕???#
# %%
loss_function_1 = nn.CrossEntropyLoss()
loss_function_2 = nn.MSELoss()
loss_function_3 = nn.MultiLabelSoftMarginLoss()
loss_function_4 = nn.MultiMarginLoss()
# %%
n_features = 6

model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adam(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# 
#
# %%
train_model(model_best, 'CEloss', loss_function_1, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.8034 Acc: 0.2179
# val Total Loss: 1.7040 Acc: 0.3376
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6990 Acc: 0.3045
# val Total Loss: 1.6106 Acc: 0.4430
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5850 Acc: 0.3869
# val Total Loss: 1.4929 Acc: 0.5063
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.4202 Acc: 0.4721
# val Total Loss: 1.3470 Acc: 0.5570
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.2608 Acc: 0.5642
# val Total Loss: 1.2121 Acc: 0.5865
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.1237 Acc: 0.6355
# val Total Loss: 1.1109 Acc: 0.6414
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.0558 Acc: 0.6397
# val Total Loss: 1.0447 Acc: 0.6414
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.9261 Acc: 0.6983
# val Total Loss: 0.9704 Acc: 0.6329
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.8581 Acc: 0.7346
# val Total Loss: 0.9156 Acc: 0.6793
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.7568 Acc: 0.7626
# val Total Loss: 0.8833 Acc: 0.6793
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.6533 Acc: 0.7933
# val Total Loss: 0.8529 Acc: 0.6920
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.6434 Acc: 0.7905
# val Total Loss: 0.8146 Acc: 0.7089
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.5292 Acc: 0.8464
# val Total Loss: 0.8111 Acc: 0.7215
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.4674 Acc: 0.8534
# val Total Loss: 0.7994 Acc: 0.7384
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.4249 Acc: 0.8729
# val Total Loss: 0.8239 Acc: 0.7342
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.3643 Acc: 0.8911
# val Total Loss: 0.8138 Acc: 0.7300
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.3390 Acc: 0.9092
# val Total Loss: 0.8656 Acc: 0.7215
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.2976 Acc: 0.9218
# val Total Loss: 0.8326 Acc: 0.7300
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.2823 Acc: 0.9344
# val Total Loss: 0.8180 Acc: 0.7384
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.2305 Acc: 0.9413
# val Total Loss: 0.8209 Acc: 0.7384
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.2256 Acc: 0.9385
# val Total Loss: 0.8295 Acc: 0.7468
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.2134 Acc: 0.9483
# val Total Loss: 0.8393 Acc: 0.7426
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.2273 Acc: 0.9358
# val Total Loss: 0.8500 Acc: 0.7257
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.1987 Acc: 0.9567
# val Total Loss: 0.8626 Acc: 0.7089
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.1497 Acc: 0.9679
# val Total Loss: 0.8234 Acc: 0.7553
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.1484 Acc: 0.9665
# val Total Loss: 0.8283 Acc: 0.7595
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.1366 Acc: 0.9637
# val Total Loss: 0.8191 Acc: 0.7722
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.1673 Acc: 0.9553
# val Total Loss: 0.8130 Acc: 0.7511
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.1246 Acc: 0.9679
# val Total Loss: 0.8316 Acc: 0.7553
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.1161 Acc: 0.9777
# val Total Loss: 0.8282 Acc: 0.7426
# 
# Best val Acc: 0.7722
# 
#
# %%
from torch.nn.functional import one_hot

def train_model_one(model, model_name, loss_function, optimizer, scheduler, max_epochs):
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

                        # ?덉씠釉붿쓣 ?????몄퐫?⑺븯???먯떎 ?⑥닔???꾨떖
                        labels_one_hot = one_hot(labels, num_classes=n_features).float()
                        loss = loss_function(outputs, labels_one_hot)

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
# %% [markdown]
# 
# -> ?낅젰???ш린媛 [16,6]?대?濡??덉씠釉붿쓣 ?대떦 ?대옒???섎줈 ?몄퐫?⑺빐?쇳븿 '?먰빂 ?몄퐫?? 異붽?
#
# %%
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adam(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# 
#
# %%
train_model_one(model_best, 'MSELoss', loss_function_2, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 0.3047 Acc: 0.1941
# val Total Loss: 0.1709 Acc: 0.2321
# 
# Epoch 2/30
# ----------
# train Total Loss: 0.2430 Acc: 0.2430
# val Total Loss: 0.1499 Acc: 0.3165
# 
# Epoch 3/30
# ----------
# train Total Loss: 0.2452 Acc: 0.2346
# val Total Loss: 0.1372 Acc: 0.3840
# 
# Epoch 4/30
# ----------
# train Total Loss: 0.2212 Acc: 0.2891
# val Total Loss: 0.1297 Acc: 0.4219
# 
# Epoch 5/30
# ----------
# train Total Loss: 0.2102 Acc: 0.3436
# val Total Loss: 0.1235 Acc: 0.4515
# 
# Epoch 6/30
# ----------
# train Total Loss: 0.2017 Acc: 0.3729
# val Total Loss: 0.1174 Acc: 0.4979
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.1985 Acc: 0.3659
# val Total Loss: 0.1128 Acc: 0.5274
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.1800 Acc: 0.4316
# val Total Loss: 0.1111 Acc: 0.5274
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.1689 Acc: 0.4204
# val Total Loss: 0.1071 Acc: 0.5527
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.1581 Acc: 0.4944
# val Total Loss: 0.1071 Acc: 0.5654
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.1596 Acc: 0.4679
# val Total Loss: 0.1023 Acc: 0.5696
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.1512 Acc: 0.4749
# val Total Loss: 0.1014 Acc: 0.5907
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.1446 Acc: 0.4791
# val Total Loss: 0.1017 Acc: 0.5992
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.1363 Acc: 0.5000
# val Total Loss: 0.0994 Acc: 0.6034
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.1338 Acc: 0.5112
# val Total Loss: 0.0963 Acc: 0.6287
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.1242 Acc: 0.5726
# val Total Loss: 0.0981 Acc: 0.6118
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.1182 Acc: 0.5852
# val Total Loss: 0.0940 Acc: 0.6076
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.1162 Acc: 0.6145
# val Total Loss: 0.0921 Acc: 0.6287
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.1075 Acc: 0.6355
# val Total Loss: 0.0904 Acc: 0.6329
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.1051 Acc: 0.6550
# val Total Loss: 0.0872 Acc: 0.6371
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.1012 Acc: 0.6592
# val Total Loss: 0.0855 Acc: 0.6371
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.0963 Acc: 0.6899
# val Total Loss: 0.0872 Acc: 0.6498
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.0911 Acc: 0.7193
# val Total Loss: 0.0831 Acc: 0.6540
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.0907 Acc: 0.7193
# val Total Loss: 0.0804 Acc: 0.6624
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.0856 Acc: 0.7626
# val Total Loss: 0.0816 Acc: 0.6624
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.0803 Acc: 0.7654
# val Total Loss: 0.0791 Acc: 0.6962
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.0839 Acc: 0.7584
# val Total Loss: 0.0786 Acc: 0.6878
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.0759 Acc: 0.7989
# val Total Loss: 0.0793 Acc: 0.6709
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.0745 Acc: 0.7905
# val Total Loss: 0.0806 Acc: 0.6709
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.0720 Acc: 0.8142
# val Total Loss: 0.0780 Acc: 0.6878
# 
# Best val Acc: 0.6962
# 
#
# %%
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adam(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# 
#
# %%
train_model_one(model_best, 'MultiLabelSoftMarginLoss', loss_function_3, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 0.6508 Acc: 0.1885
# val Total Loss: 0.5853 Acc: 0.2785
# 
# Epoch 2/30
# ----------
# train Total Loss: 0.5488 Acc: 0.2458
# val Total Loss: 0.5110 Acc: 0.3418
# 
# Epoch 3/30
# ----------
# train Total Loss: 0.4818 Acc: 0.3184
# val Total Loss: 0.4589 Acc: 0.4304
# 
# Epoch 4/30
# ----------
# train Total Loss: 0.4342 Acc: 0.3980
# val Total Loss: 0.4166 Acc: 0.5148
# 
# Epoch 5/30
# ----------
# train Total Loss: 0.3968 Acc: 0.5056
# val Total Loss: 0.3811 Acc: 0.5907
# 
# Epoch 6/30
# ----------
# train Total Loss: 0.3650 Acc: 0.5866
# val Total Loss: 0.3516 Acc: 0.6287
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.3300 Acc: 0.6271
# val Total Loss: 0.3233 Acc: 0.6371
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.3074 Acc: 0.6578
# val Total Loss: 0.3002 Acc: 0.6540
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.2819 Acc: 0.7179
# val Total Loss: 0.2872 Acc: 0.6624
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.2617 Acc: 0.7402
# val Total Loss: 0.2687 Acc: 0.6793
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.2374 Acc: 0.7835
# val Total Loss: 0.2553 Acc: 0.6962
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.2178 Acc: 0.7863
# val Total Loss: 0.2461 Acc: 0.7300
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.1986 Acc: 0.8226
# val Total Loss: 0.2400 Acc: 0.7004
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.1882 Acc: 0.8184
# val Total Loss: 0.2386 Acc: 0.7004
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.1609 Acc: 0.8631
# val Total Loss: 0.2267 Acc: 0.7426
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.1473 Acc: 0.8743
# val Total Loss: 0.2265 Acc: 0.7342
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.1381 Acc: 0.9036
# val Total Loss: 0.2266 Acc: 0.7173
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.1280 Acc: 0.9050
# val Total Loss: 0.2276 Acc: 0.7215
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.1137 Acc: 0.9190
# val Total Loss: 0.2200 Acc: 0.7384
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.1000 Acc: 0.9385
# val Total Loss: 0.2241 Acc: 0.7300
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.0949 Acc: 0.9344
# val Total Loss: 0.2328 Acc: 0.7468
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.0901 Acc: 0.9441
# val Total Loss: 0.2334 Acc: 0.7300
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.0863 Acc: 0.9413
# val Total Loss: 0.2250 Acc: 0.7426
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.0744 Acc: 0.9637
# val Total Loss: 0.2274 Acc: 0.7215
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.0654 Acc: 0.9679
# val Total Loss: 0.2292 Acc: 0.7511
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.0674 Acc: 0.9595
# val Total Loss: 0.2250 Acc: 0.7215
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.0581 Acc: 0.9637
# val Total Loss: 0.2394 Acc: 0.7300
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.0628 Acc: 0.9581
# val Total Loss: 0.2341 Acc: 0.7300
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.0604 Acc: 0.9637
# val Total Loss: 0.2385 Acc: 0.7131
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.0587 Acc: 0.9651
# val Total Loss: 0.2404 Acc: 0.7426
# 
# Best val Acc: 0.7511
# 
#
# %%
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adam(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_best, 'MultiMarginLoss', loss_function_4, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 0.7744 Acc: 0.2137
# val Total Loss: 0.7144 Acc: 0.2489
# 
# Epoch 2/30
# ----------
# train Total Loss: 0.6704 Acc: 0.2584
# val Total Loss: 0.6137 Acc: 0.3249
# 
# Epoch 3/30
# ----------
# train Total Loss: 0.5727 Acc: 0.3450
# val Total Loss: 0.5256 Acc: 0.4051
# 
# Epoch 4/30
# ----------
# train Total Loss: 0.4968 Acc: 0.4497
# val Total Loss: 0.4545 Acc: 0.5063
# 
# Epoch 5/30
# ----------
# train Total Loss: 0.3759 Acc: 0.5405
# val Total Loss: 0.3910 Acc: 0.5907
# 
# Epoch 6/30
# ----------
# train Total Loss: 0.3464 Acc: 0.5615
# val Total Loss: 0.3533 Acc: 0.6203
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.3074 Acc: 0.5992
# val Total Loss: 0.3244 Acc: 0.6456
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.2829 Acc: 0.6453
# val Total Loss: 0.2985 Acc: 0.6624
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.2290 Acc: 0.6941
# val Total Loss: 0.2846 Acc: 0.6878
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.2122 Acc: 0.6955
# val Total Loss: 0.2766 Acc: 0.7131
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.1938 Acc: 0.7304
# val Total Loss: 0.2706 Acc: 0.7131
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.1659 Acc: 0.7500
# val Total Loss: 0.2568 Acc: 0.6962
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.1577 Acc: 0.7486
# val Total Loss: 0.2461 Acc: 0.7384
# 
# Epoch 14/30
# ----------
# val Total Loss: 0.2421 Acc: 0.7426
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.1229 Acc: 0.8059
# val Total Loss: 0.2501 Acc: 0.7342
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.1084 Acc: 0.8450
# val Total Loss: 0.2371 Acc: 0.7342
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.1083 Acc: 0.8338
# val Total Loss: 0.2387 Acc: 0.7384
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.0901 Acc: 0.8561
# val Total Loss: 0.2264 Acc: 0.7468
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.0694 Acc: 0.8883
# val Total Loss: 0.2319 Acc: 0.7300
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.0628 Acc: 0.8911
# val Total Loss: 0.2191 Acc: 0.7426
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.0618 Acc: 0.9022
# val Total Loss: 0.2207 Acc: 0.7384
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.0542 Acc: 0.9218
# val Total Loss: 0.2306 Acc: 0.7300
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.0533 Acc: 0.9092
# val Total Loss: 0.2328 Acc: 0.7426
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.0439 Acc: 0.9232
# val Total Loss: 0.2393 Acc: 0.7257
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.0353 Acc: 0.9483
# val Total Loss: 0.2281 Acc: 0.7553
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.0482 Acc: 0.9148
# val Total Loss: 0.2338 Acc: 0.7342
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.0398 Acc: 0.9385
# val Total Loss: 0.2343 Acc: 0.7384
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.0465 Acc: 0.9008
# val Total Loss: 0.2285 Acc: 0.7300
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.0355 Acc: 0.9455
# val Total Loss: 0.2366 Acc: 0.7173
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.0251 Acc: 0.9623
# val Total Loss: 0.2237 Acc: 0.7426
# 
# Best val Acc: 0.7553
# 
#
# %%
import torch
import matplotlib.pyplot as plt

ce_acc_dict = torch.load("CEloss_acc_dict.pth")
multi_label_soft_margin_loss_acc_dict = torch.load("MultiLabelSoftMarginLoss_acc_dict.pth")
multi_margin_loss_acc_dict = torch.load("MultiMarginLoss_acc_dict.pth")
mse_loss_acc_dict = torch.load("MSELoss_acc_dict.pth")

plt.figure(figsize=(12, 8))

plt.plot(range(len(ce_acc_dict['val'])), ce_acc_dict['val'], label='CELoss Validation Accuracy', color='blue')
plt.plot(range(len(multi_label_soft_margin_loss_acc_dict['val'])), multi_label_soft_margin_loss_acc_dict['val'], label='MultiLabelSoftMarginLoss', color='orange')
plt.plot(range(len(multi_margin_loss_acc_dict['val'])), multi_margin_loss_acc_dict['val'], label='MultiMarginLoss', color='green')
plt.plot(range(len(mse_loss_acc_dict['val'])), mse_loss_acc_dict['val'], label='MSELoss', color='red')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy of Loss Functions')
plt.legend()
plt.grid(True)

plt.show()
# %% [markdown]
# Saved output
# <Figure size 1200x800 with 1 Axes>
#
# %%
loss_function_best=loss_function_1
# %% [markdown]
# 
# ### **3. Optimizer ?좎젙**
#
# %% [markdown]
# 
# **[Optimizer]**
# 
# Adam : gradient??1李?諛?2李?moment??異붿젙移섎? 湲곕컲?쇰줈 ?숈뒿瑜?怨꾩궛 / adaptive learning rater瑜?湲곕컲?쇰줈 step size 議곗젙
# 
# Adagrad : 怨쇨굅 gradient瑜?湲곕컲?쇰줈 ?숈뒿瑜?議곗젙
# 
# Rmsprop : grad^2???대룞 ?됯퇏???댁슜?섏뿬 ?숈뒿瑜?議곗젅
# 
# SGD : 湲곕낯 / 媛곴컖 parameter??learning rate 怨깊븿
# 
# Ranger : Rectified Adam + Lookahead
#
# %%
!pip install torch_optimizer
# %% [markdown]
# Saved output
# Collecting torch_optimizer
#   Downloading torch_optimizer-0.3.0-py3-none-any.whl (61 kB)
# [?25l     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m0.0/61.9 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m61.9/61.9 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
# [?25hRequirement already satisfied: torch>=1.5.0 in /usr/local/lib/python3.10/dist-packages (from torch_optimizer) (2.3.0+cu121)
# Collecting pytorch-ranger>=0.1.1 (from torch_optimizer)
#   Downloading pytorch_ranger-0.1.1-py3-none-any.whl (14 kB)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (3.14.0)
# Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (4.11.0)
# Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (1.12)
# Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (3.3)
# Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (3.1.4)
# Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (2023.6.0)
# Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
# Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
# Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
# Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
# Collecting nvidia-cublas-cu12==12.1.3.1 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
# Collecting nvidia-cufft-cu12==11.0.2.54 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
# Collecting nvidia-curand-cu12==10.3.2.106 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
# Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
# Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
# Collecting nvidia-nccl-cu12==2.20.5 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
# Collecting nvidia-nvtx-cu12==12.1.105 (from torch>=1.5.0->torch_optimizer)
#   Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
# Requirement already satisfied: triton==2.3.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.5.0->torch_optimizer) (2.3.0)
# Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.5.0->torch_optimizer)
#   Downloading nvidia_nvjitlink_cu12-12.5.40-py3-none-manylinux2014_x86_64.whl (21.3 MB)
# [2K     [90m?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺?곣봺[0m [32m21.3/21.3 MB[0m [31m80.7 MB/s[0m eta [36m0:00:00[0m
# [?25hRequirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.5.0->torch_optimizer) (2.1.5)
# Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.5.0->torch_optimizer) (1.3.0)
# Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, pytorch-ranger, torch_optimizer
# Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.5.40 nvidia-nvtx-cu12-12.1.105 pytorch-ranger-0.1.1 torch_optimizer-0.3.0
# 
#
# %%
#optimizer1 : adam

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adam(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# 
#
# %%
train_model(model_best, 'Adam', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.7982 Acc: 0.1955
# val Total Loss: 1.6792 Acc: 0.3122
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.6775 Acc: 0.2933
# val Total Loss: 1.5771 Acc: 0.4388
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.5279 Acc: 0.4134
# val Total Loss: 1.4554 Acc: 0.5232
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.3964 Acc: 0.5070
# val Total Loss: 1.3229 Acc: 0.5485
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.2505 Acc: 0.5559
# val Total Loss: 1.2255 Acc: 0.5781
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.1380 Acc: 0.6034
# val Total Loss: 1.1321 Acc: 0.6160
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.0169 Acc: 0.6592
# val Total Loss: 1.0407 Acc: 0.6456
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.9153 Acc: 0.7067
# val Total Loss: 0.9853 Acc: 0.6456
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.8204 Acc: 0.7458
# val Total Loss: 0.9412 Acc: 0.6582
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.7339 Acc: 0.7612
# val Total Loss: 0.8912 Acc: 0.6878
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.6650 Acc: 0.7877
# val Total Loss: 0.8429 Acc: 0.7468
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.6094 Acc: 0.8045
# val Total Loss: 0.8490 Acc: 0.7046
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.5385 Acc: 0.8450
# val Total Loss: 0.8413 Acc: 0.7131
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.4757 Acc: 0.8575
# val Total Loss: 0.8139 Acc: 0.7131
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.4548 Acc: 0.8617
# val Total Loss: 0.8143 Acc: 0.7131
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.3711 Acc: 0.8869
# val Total Loss: 0.8072 Acc: 0.7426
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.3404 Acc: 0.9134
# val Total Loss: 0.8183 Acc: 0.7384
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.3389 Acc: 0.9064
# val Total Loss: 0.8134 Acc: 0.7257
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.2908 Acc: 0.9274
# val Total Loss: 0.8178 Acc: 0.7257
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.2684 Acc: 0.9176
# val Total Loss: 0.8136 Acc: 0.7173
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.2384 Acc: 0.9399
# val Total Loss: 0.8435 Acc: 0.7131
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.2302 Acc: 0.9385
# val Total Loss: 0.8216 Acc: 0.7426
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.1715 Acc: 0.9665
# val Total Loss: 0.7989 Acc: 0.7511
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.1629 Acc: 0.9553
# val Total Loss: 0.8085 Acc: 0.7511
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.1680 Acc: 0.9665
# val Total Loss: 0.8419 Acc: 0.7426
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.1511 Acc: 0.9679
# val Total Loss: 0.8646 Acc: 0.7215
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.1505 Acc: 0.9623
# val Total Loss: 0.8515 Acc: 0.7215
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.1189 Acc: 0.9791
# val Total Loss: 0.8393 Acc: 0.7257
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.1278 Acc: 0.9651
# val Total Loss: 0.8647 Acc: 0.7300
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.1236 Acc: 0.9665
# val Total Loss: 0.8954 Acc: 0.7215
# 
# Best val Acc: 0.7511
# 
#
# %%
#optimizer2 : Adagrad

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.Adagrad(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_best, 'Adagrad', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.8542 Acc: 0.1690
# val Total Loss: 1.7986 Acc: 0.2068
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.8090 Acc: 0.1983
# val Total Loss: 1.7864 Acc: 0.2152
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.8267 Acc: 0.1844
# val Total Loss: 1.7702 Acc: 0.2278
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.8057 Acc: 0.1858
# val Total Loss: 1.7649 Acc: 0.2321
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.7948 Acc: 0.2067
# val Total Loss: 1.7577 Acc: 0.2363
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.7729 Acc: 0.2277
# val Total Loss: 1.7509 Acc: 0.2405
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.7796 Acc: 0.2291
# val Total Loss: 1.7479 Acc: 0.2363
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.7573 Acc: 0.2444
# val Total Loss: 1.7384 Acc: 0.2574
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.7642 Acc: 0.2193
# val Total Loss: 1.7316 Acc: 0.2616
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.7334 Acc: 0.2737
# val Total Loss: 1.7258 Acc: 0.2574
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.7478 Acc: 0.2500
# val Total Loss: 1.7213 Acc: 0.2658
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.7534 Acc: 0.2626
# val Total Loss: 1.7161 Acc: 0.2574
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.7330 Acc: 0.2765
# val Total Loss: 1.7113 Acc: 0.2743
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.7504 Acc: 0.2332
# val Total Loss: 1.7068 Acc: 0.2658
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.7141 Acc: 0.2654
# val Total Loss: 1.7083 Acc: 0.2658
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.7423 Acc: 0.2514
# val Total Loss: 1.6993 Acc: 0.2743
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.7052 Acc: 0.2877
# val Total Loss: 1.7012 Acc: 0.2743
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.7277 Acc: 0.2751
# val Total Loss: 1.6933 Acc: 0.2743
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.6756 Acc: 0.3017
# val Total Loss: 1.6911 Acc: 0.2911
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.6956 Acc: 0.2919
# val Total Loss: 1.6865 Acc: 0.2827
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.6803 Acc: 0.3170
# val Total Loss: 1.6812 Acc: 0.2954
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.6776 Acc: 0.2891
# val Total Loss: 1.6787 Acc: 0.2827
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.6806 Acc: 0.2947
# val Total Loss: 1.6753 Acc: 0.3038
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.7128 Acc: 0.2779
# val Total Loss: 1.6734 Acc: 0.2954
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.6843 Acc: 0.2961
# val Total Loss: 1.6665 Acc: 0.3080
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.7034 Acc: 0.2598
# val Total Loss: 1.6641 Acc: 0.3080
# 
# Epoch 27/30
# ----------
# train Total Loss: 1.6485 Acc: 0.3254
# val Total Loss: 1.6626 Acc: 0.2996
# 
# Epoch 28/30
# ----------
# train Total Loss: 1.6714 Acc: 0.3226
# val Total Loss: 1.6579 Acc: 0.3080
# 
# Epoch 29/30
# ----------
# train Total Loss: 1.6393 Acc: 0.3240
# val Total Loss: 1.6553 Acc: 0.3207
# 
# Epoch 30/30
# ----------
# train Total Loss: 1.6274 Acc: 0.3380
# val Total Loss: 1.6514 Acc: 0.3333
# 
# Best val Acc: 0.3333
# 
#
# %%
#optimizer3 : RMSprop

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.RMSprop(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_best, 'RMSprop', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.7000 Acc: 0.3017
# val Total Loss: 1.5218 Acc: 0.4557
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.4129 Acc: 0.4888
# val Total Loss: 1.3145 Acc: 0.5359
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.2176 Acc: 0.5922
# val Total Loss: 1.1756 Acc: 0.5949
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.0947 Acc: 0.6271
# val Total Loss: 1.0721 Acc: 0.6414
# 
# Epoch 5/30
# ----------
# train Total Loss: 0.9593 Acc: 0.6955
# val Total Loss: 0.9979 Acc: 0.6878
# 
# Epoch 6/30
# ----------
# train Total Loss: 0.8710 Acc: 0.7207
# val Total Loss: 0.9413 Acc: 0.6835
# 
# Epoch 7/30
# ----------
# train Total Loss: 0.7878 Acc: 0.7514
# val Total Loss: 0.9340 Acc: 0.7089
# 
# Epoch 8/30
# ----------
# train Total Loss: 0.7130 Acc: 0.7696
# val Total Loss: 0.8887 Acc: 0.7173
# 
# Epoch 9/30
# ----------
# train Total Loss: 0.5852 Acc: 0.8170
# val Total Loss: 0.8617 Acc: 0.7215
# 
# Epoch 10/30
# ----------
# train Total Loss: 0.5675 Acc: 0.8254
# val Total Loss: 0.8068 Acc: 0.7426
# 
# Epoch 11/30
# ----------
# train Total Loss: 0.4751 Acc: 0.8589
# val Total Loss: 0.8172 Acc: 0.7342
# 
# Epoch 12/30
# ----------
# train Total Loss: 0.4217 Acc: 0.8855
# val Total Loss: 0.7963 Acc: 0.7257
# 
# Epoch 13/30
# ----------
# train Total Loss: 0.3688 Acc: 0.8953
# val Total Loss: 0.8010 Acc: 0.7468
# 
# Epoch 14/30
# ----------
# train Total Loss: 0.3536 Acc: 0.9050
# val Total Loss: 0.8098 Acc: 0.7131
# 
# Epoch 15/30
# ----------
# train Total Loss: 0.3273 Acc: 0.9008
# val Total Loss: 0.7939 Acc: 0.7511
# 
# Epoch 16/30
# ----------
# train Total Loss: 0.2681 Acc: 0.9274
# val Total Loss: 0.8305 Acc: 0.7511
# 
# Epoch 17/30
# ----------
# train Total Loss: 0.2643 Acc: 0.9218
# val Total Loss: 0.8669 Acc: 0.7300
# 
# Epoch 18/30
# ----------
# train Total Loss: 0.2354 Acc: 0.9274
# val Total Loss: 0.8449 Acc: 0.7300
# 
# Epoch 19/30
# ----------
# train Total Loss: 0.1925 Acc: 0.9483
# val Total Loss: 0.8552 Acc: 0.7384
# 
# Epoch 20/30
# ----------
# train Total Loss: 0.1539 Acc: 0.9679
# val Total Loss: 0.8701 Acc: 0.7468
# 
# Epoch 21/30
# ----------
# train Total Loss: 0.1863 Acc: 0.9469
# val Total Loss: 0.8974 Acc: 0.7257
# 
# Epoch 22/30
# ----------
# train Total Loss: 0.1655 Acc: 0.9553
# val Total Loss: 0.8886 Acc: 0.7215
# 
# Epoch 23/30
# ----------
# train Total Loss: 0.1268 Acc: 0.9665
# val Total Loss: 0.9121 Acc: 0.7173
# 
# Epoch 24/30
# ----------
# train Total Loss: 0.1287 Acc: 0.9623
# val Total Loss: 0.8805 Acc: 0.7553
# 
# Epoch 25/30
# ----------
# train Total Loss: 0.1524 Acc: 0.9581
# val Total Loss: 0.9105 Acc: 0.7384
# 
# Epoch 26/30
# ----------
# train Total Loss: 0.1228 Acc: 0.9623
# val Total Loss: 0.9618 Acc: 0.7089
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.1200 Acc: 0.9623
# val Total Loss: 0.9186 Acc: 0.7215
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.1108 Acc: 0.9735
# val Total Loss: 0.9398 Acc: 0.7342
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.1192 Acc: 0.9665
# val Total Loss: 0.9353 Acc: 0.7173
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.1164 Acc: 0.9651
# val Total Loss: 0.9210 Acc: 0.7300
# 
# Best val Acc: 0.7553
# 
#
# %%
#optimizer4 : SGD

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.SGD(model_best.parameters(), lr=1e-5, momentum=0.9) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_best, 'SGD', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# train Total Loss: 1.8688 Acc: 0.1564
# val Total Loss: 1.7974 Acc: 0.1983
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.8541 Acc: 0.1578
# val Total Loss: 1.7883 Acc: 0.2068
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.8265 Acc: 0.1899
# val Total Loss: 1.7783 Acc: 0.2405
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.8181 Acc: 0.1844
# val Total Loss: 1.7755 Acc: 0.2321
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.8237 Acc: 0.1997
# val Total Loss: 1.7665 Acc: 0.2532
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.8137 Acc: 0.1899
# val Total Loss: 1.7631 Acc: 0.2616
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.7989 Acc: 0.1997
# val Total Loss: 1.7553 Acc: 0.2785
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.7999 Acc: 0.2235
# val Total Loss: 1.7515 Acc: 0.2658
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.8061 Acc: 0.2039
# val Total Loss: 1.7502 Acc: 0.2616
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.8022 Acc: 0.1941
# val Total Loss: 1.7432 Acc: 0.2700
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.7842 Acc: 0.2221
# val Total Loss: 1.7384 Acc: 0.2743
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.7610 Acc: 0.2500
# val Total Loss: 1.7338 Acc: 0.2785
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.7693 Acc: 0.2388
# val Total Loss: 1.7332 Acc: 0.2911
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.7601 Acc: 0.2416
# val Total Loss: 1.7236 Acc: 0.3038
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.7685 Acc: 0.2374
# val Total Loss: 1.7219 Acc: 0.2785
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.7616 Acc: 0.2388
# val Total Loss: 1.7166 Acc: 0.3080
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.7572 Acc: 0.2374
# val Total Loss: 1.7136 Acc: 0.2996
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.7344 Acc: 0.2444
# val Total Loss: 1.7057 Acc: 0.3207
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.7206 Acc: 0.2737
# val Total Loss: 1.7037 Acc: 0.3165
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.7177 Acc: 0.2486
# val Total Loss: 1.7004 Acc: 0.3080
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.7338 Acc: 0.2640
# val Total Loss: 1.6974 Acc: 0.3207
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.7264 Acc: 0.2723
# val Total Loss: 1.6887 Acc: 0.3291
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.7211 Acc: 0.2709
# val Total Loss: 1.6894 Acc: 0.3460
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.7219 Acc: 0.2821
# val Total Loss: 1.6826 Acc: 0.3122
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.6919 Acc: 0.2821
# val Total Loss: 1.6825 Acc: 0.3333
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.7198 Acc: 0.2570
# val Total Loss: 1.6752 Acc: 0.3418
# 
# Epoch 27/30
# ----------
# train Total Loss: 1.7047 Acc: 0.2654
# val Total Loss: 1.6730 Acc: 0.3502
# 
# Epoch 28/30
# ----------
# train Total Loss: 1.6708 Acc: 0.2835
# val Total Loss: 1.6644 Acc: 0.3629
# 
# Epoch 29/30
# ----------
# train Total Loss: 1.7037 Acc: 0.2975
# val Total Loss: 1.6604 Acc: 0.3544
# 
# Epoch 30/30
# ----------
# train Total Loss: 1.7023 Acc: 0.2765
# val Total Loss: 1.6576 Acc: 0.3671
# 
# Best val Acc: 0.3671
# 
#
# %%
#optimizer5 : Ranger

from torch_optimizer import Ranger

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = Ranger(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
# %%
train_model(model_best, 'Ranger', loss_function_best, optimizer, scheduler4, max_epochs)
# %% [markdown]
# Saved output
# Epoch 1/30
# ----------
# 
# /usr/local/lib/python3.10/dist-packages/pytorch_ranger/ranger.py:172: UserWarning: This overload of addcmul_ is deprecated:
# 	addcmul_(Number value, Tensor tensor1, Tensor tensor2)
# Consider using one of the following signatures instead:
# 	addcmul_(Tensor tensor1, Tensor tensor2, *, Number value) (Triggered internally at ../torch/csrc/utils/python_arg_parser.cpp:1578.)
#   exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
# 
# train Total Loss: 1.9239 Acc: 0.1606
# val Total Loss: 1.8629 Acc: 0.1814
# 
# Epoch 2/30
# ----------
# train Total Loss: 1.9098 Acc: 0.1648
# val Total Loss: 1.8504 Acc: 0.1814
# 
# Epoch 3/30
# ----------
# train Total Loss: 1.8817 Acc: 0.1676
# val Total Loss: 1.8287 Acc: 0.1857
# 
# Epoch 4/30
# ----------
# train Total Loss: 1.8620 Acc: 0.1760
# val Total Loss: 1.8067 Acc: 0.1814
# 
# Epoch 5/30
# ----------
# train Total Loss: 1.8441 Acc: 0.1788
# val Total Loss: 1.7841 Acc: 0.1899
# 
# Epoch 6/30
# ----------
# train Total Loss: 1.8105 Acc: 0.1746
# val Total Loss: 1.7646 Acc: 0.1983
# 
# Epoch 7/30
# ----------
# train Total Loss: 1.7876 Acc: 0.2207
# val Total Loss: 1.7357 Acc: 0.2110
# 
# Epoch 8/30
# ----------
# train Total Loss: 1.7482 Acc: 0.2416
# val Total Loss: 1.7196 Acc: 0.2236
# 
# Epoch 9/30
# ----------
# train Total Loss: 1.7287 Acc: 0.2430
# val Total Loss: 1.6911 Acc: 0.2447
# 
# Epoch 10/30
# ----------
# train Total Loss: 1.6811 Acc: 0.2765
# val Total Loss: 1.6651 Acc: 0.2700
# 
# Epoch 11/30
# ----------
# train Total Loss: 1.6633 Acc: 0.2947
# val Total Loss: 1.6343 Acc: 0.3165
# 
# Epoch 12/30
# ----------
# train Total Loss: 1.6406 Acc: 0.3338
# val Total Loss: 1.6146 Acc: 0.3249
# 
# Epoch 13/30
# ----------
# train Total Loss: 1.5941 Acc: 0.3575
# val Total Loss: 1.5834 Acc: 0.3924
# 
# Epoch 14/30
# ----------
# train Total Loss: 1.5515 Acc: 0.3841
# val Total Loss: 1.5510 Acc: 0.4051
# 
# Epoch 15/30
# ----------
# train Total Loss: 1.5149 Acc: 0.4274
# val Total Loss: 1.5219 Acc: 0.4135
# 
# Epoch 16/30
# ----------
# train Total Loss: 1.4992 Acc: 0.4092
# val Total Loss: 1.4807 Acc: 0.4557
# 
# Epoch 17/30
# ----------
# train Total Loss: 1.4355 Acc: 0.4679
# val Total Loss: 1.4481 Acc: 0.4937
# 
# Epoch 18/30
# ----------
# train Total Loss: 1.3712 Acc: 0.5209
# val Total Loss: 1.4060 Acc: 0.5359
# 
# Epoch 19/30
# ----------
# train Total Loss: 1.3542 Acc: 0.5056
# val Total Loss: 1.3614 Acc: 0.5443
# 
# Epoch 20/30
# ----------
# train Total Loss: 1.3135 Acc: 0.5489
# val Total Loss: 1.3224 Acc: 0.5696
# 
# Epoch 21/30
# ----------
# train Total Loss: 1.2626 Acc: 0.5726
# val Total Loss: 1.2802 Acc: 0.5865
# 
# Epoch 22/30
# ----------
# train Total Loss: 1.1954 Acc: 0.6229
# val Total Loss: 1.2435 Acc: 0.6034
# 
# Epoch 23/30
# ----------
# train Total Loss: 1.1529 Acc: 0.6047
# val Total Loss: 1.1980 Acc: 0.6329
# 
# Epoch 24/30
# ----------
# train Total Loss: 1.1385 Acc: 0.6159
# val Total Loss: 1.1785 Acc: 0.6456
# 
# Epoch 25/30
# ----------
# train Total Loss: 1.0895 Acc: 0.6313
# val Total Loss: 1.1498 Acc: 0.6160
# 
# Epoch 26/30
# ----------
# train Total Loss: 1.0540 Acc: 0.6439
# val Total Loss: 1.1105 Acc: 0.6371
# 
# Epoch 27/30
# ----------
# train Total Loss: 0.9892 Acc: 0.6662
# val Total Loss: 1.0801 Acc: 0.6160
# 
# Epoch 28/30
# ----------
# train Total Loss: 0.9600 Acc: 0.6983
# val Total Loss: 1.0511 Acc: 0.6414
# 
# Epoch 29/30
# ----------
# train Total Loss: 0.9403 Acc: 0.6816
# val Total Loss: 1.0390 Acc: 0.6498
# 
# Epoch 30/30
# ----------
# train Total Loss: 0.8814 Acc: 0.7179
# val Total Loss: 1.0069 Acc: 0.6624
# 
# Best val Acc: 0.6624
# 
#
# %%
import matplotlib.pyplot as plt
import torch

# ?뺥솗???뚯씪 遺덈윭?ㅺ린
adam_acc_dict = torch.load("Adam_acc_dict.pth")
adagrad_acc_dict = torch.load("Adagrad_acc_dict.pth")
rmsprop_acc_dict = torch.load("RMSprop_acc_dict.pth")
sgd_acc_dict = torch.load("SGD_acc_dict.pth")
ranger_acc_dict = torch.load("Ranger_acc_dict.pth")

# 洹몃옒??洹몃━湲?plt.figure(figsize=(12, 8))

# Validation Accuracy 洹몃옒??plt.plot(range(len(adam_acc_dict['val'])), adam_acc_dict['val'], label='Adam', color='blue')
plt.plot(range(len(adagrad_acc_dict['val'])), adagrad_acc_dict['val'], label='Adagrad', color='orange')
plt.plot(range(len(rmsprop_acc_dict['val'])), rmsprop_acc_dict['val'], label='RMSprop', color='green')
plt.plot(range(len(sgd_acc_dict['val'])), sgd_acc_dict['val'], label='SGD', color='red')
plt.plot(range(len(ranger_acc_dict['val'])), ranger_acc_dict['val'], label='Ranger', color='purple')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Comparison of Validation Accuracy for Different Optimizers')
plt.legend()
plt.grid(True)

plt.show()
# %% [markdown]
# Saved output
# <Figure size 1200x800 with 1 Axes>
#
# %%
optimizer_best=torch.optim.RMSprop(model_best.parameters(), lr=1e-5)
# %%
#理쒖쥌 best model : resnet152 + CELoss + RMSprop
##理쒖쥌 best model ?꾩껜 code

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
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.RMSprop(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_function_best= nn.CrossEntropyLoss()

train_model(model_best, 'RMSprop', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# 
# ### **4. ?좎젙??model濡?test ?대?吏 異쒕젰 諛?6媛??뺣쪧 異쒕젰(?쒕뜡 10媛?**
#
# %%
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

    #?쒕뜡?섍쾶 ?대?吏 ?좏깮
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
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

visualize_random_predictions(model_best, dataloaders['test'], class_names)  #test ?곗씠?곗뀑留뚯쓣 ?댁슜?섏뿬 ?쒓컖???섑뻾
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# 
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
# <Figure size 500x500 with 2 Axes>
#
# %% [markdown]
# 
# ### **5. ?좎젙??model濡??몃? ?대?吏 異쒕젰**
#
# %%
def load_image(image_data, transform=None):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')    #?대?吏 ?곗씠??image 媛앹껜濡?蹂??    if transform:
        image = transform(image).unsqueeze(0)   #李⑥썝 異붽?
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
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    plt.imshow(image)
    plt.title(f'Predicted: {class_names[pred_class]}')
    plt.axis('off')
    plt.show()

    print("Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {pred_probs[0][i]:.4f}")
# %%
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# %%
import io
from ipywidgets import FileUpload
from IPython.display import display

upload_widget = FileUpload(accept='image/*', multiple=False)

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

def on_upload_change(change):
    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)

upload_widget.observe(on_upload_change, names='value')

display(upload_widget)
# %% [markdown]
# Saved output
# FileUpload(value={}, accept='image/*', description='Upload')
# <Figure size 640x480 with 1 Axes>
# Prediction Probabilities:
# anger: 0.0001
# disgust: 0.0001
# fear: 0.0004
# happy: 0.0001
# pain: 0.0001
# sad: 0.9992
# 
#
# %%
import io
from ipywidgets import FileUpload
from IPython.display import display

upload_widget = FileUpload(accept='image/*', multiple=False)
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

def on_upload_change(change):
    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)

upload_widget.observe(on_upload_change, names='value')

display(upload_widget)
# %% [markdown]
# Saved output
# FileUpload(value={}, accept='image/*', description='Upload')
# <Figure size 640x480 with 1 Axes>
# Prediction Probabilities:
# anger: 0.3324
# disgust: 0.1897
# fear: 0.1084
# happy: 0.1058
# pain: 0.0970
# sad: 0.1667
# 
#
# %%
import io
from ipywidgets import FileUpload
from IPython.display import display

upload_widget = FileUpload(accept='image/*', multiple=False)
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

def on_upload_change(change):
    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)

upload_widget.observe(on_upload_change, names='value')

display(upload_widget)
# %% [markdown]
# Saved output
# FileUpload(value={}, accept='image/*', description='Upload')
# <Figure size 640x480 with 1 Axes>
# Prediction Probabilities:
# anger: 0.2246
# disgust: 0.2581
# fear: 0.0675
# happy: 0.1556
# pain: 0.0901
# sad: 0.2042
# 
#
# %%
import io
from ipywidgets import FileUpload
from IPython.display import display

upload_widget = FileUpload(accept='image/*', multiple=False)
n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)

model_best.load_state_dict(torch.load('RMSprop_model_params.pt'))

def on_upload_change(change):
    image_data = next(iter(change['new'].values()))['content']
    pred_class, pred_probs = predict_image(model_best, image_data, transform, class_names)
    visualize_prediction(image_data, pred_class, pred_probs, class_names)

upload_widget.observe(on_upload_change, names='value')

display(upload_widget)
# %% [markdown]
# Saved output
# FileUpload(value={}, accept='image/*', description='Upload')
# <Figure size 640x480 with 1 Axes>
# Prediction Probabilities:
# anger: 0.1338
# disgust: 0.2183
# fear: 0.0882
# happy: 0.2527
# pain: 0.0689
# sad: 0.2381
# 
#
# %% [markdown]
# 
# ### **6. (異붽?) ?좎젙??model epoch=100 ?숈뒿**
#
# %%
#理쒖쥌 best model : resnet152 + CELoss + RMSprop
##理쒖쥌 best model ?꾩껜 code

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

max_epochs = 100

n_features = 6
model_best = models.resnet152(pretrained=True)
model_best.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout
    nn.Linear(model_best.fc.in_features, n_features)  # 留덉?留?fully connected layer
)
model_best.to(device)
optimizer = torch.optim.RMSprop(model_best.parameters(), lr=1e-5) # optimizer ?뺤쓽
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_function_best= nn.CrossEntropyLoss()

train_model(model_best, 'RMSprop', loss_function_best, optimizer, scheduler, max_epochs)
# %% [markdown]
# 
# #### **??肄붾뱶(epoch=100) 異쒕젰寃곌낵**
# 
# 湲곗〈 colab ?뚯씪???꾨땶 ???뚯씪?먯꽌 ?ㅽ뻾?섏???# -> ?ㅽ뻾寃곌낵瑜??곕줈 泥⑤???# 
#
# %% [markdown]
# 
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet152_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet152_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/resnet152-394f9c45.pth" to /root/.cache/torch/hub/checkpoints/resnet152-394f9c45.pth
# 100%|?댿뻽?댿뻽?댿뻽?댿뻽?댿뻽| 230M/230M [00:04<00:00, 58.7MB/s]
# Epoch 1/100
# ----------
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# train Total Loss: 1.7306 Acc: 0.2696
# val Total Loss: 1.6233 Acc: 0.4008
# 
# Epoch 2/100
# ----------
# train Total Loss: 1.4524 Acc: 0.4623
# val Total Loss: 1.4128 Acc: 0.5401
# 
# Epoch 3/100
# ----------
# train Total Loss: 1.2518 Acc: 0.5768
# val Total Loss: 1.2835 Acc: 0.5359
# 
# Epoch 4/100
# ----------
# train Total Loss: 1.1351 Acc: 0.6187
# val Total Loss: 1.1738 Acc: 0.5865
# 
# Epoch 5/100
# ----------
# train Total Loss: 1.0104 Acc: 0.6522
# val Total Loss: 1.0876 Acc: 0.5992
# 
# Epoch 6/100
# ----------
# train Total Loss: 0.9244 Acc: 0.6885
# val Total Loss: 1.0084 Acc: 0.6456
# 
# Epoch 7/100
# ----------
# train Total Loss: 0.8253 Acc: 0.7402
# val Total Loss: 0.9754 Acc: 0.6540
# 
# Epoch 8/100
# ----------
# train Total Loss: 0.7384 Acc: 0.7640
# val Total Loss: 0.9459 Acc: 0.6414
# 
# Epoch 9/100
# ----------
# train Total Loss: 0.6731 Acc: 0.7849
# val Total Loss: 0.9069 Acc: 0.6962
# 
# Epoch 10/100
# ----------
# train Total Loss: 0.5796 Acc: 0.8338
# val Total Loss: 0.8968 Acc: 0.6835
# 
# Epoch 11/100
# ----------
# train Total Loss: 0.5556 Acc: 0.8087
# val Total Loss: 0.8741 Acc: 0.7089
# 
# Epoch 12/100
# ----------
# train Total Loss: 0.4999 Acc: 0.8520
# val Total Loss: 0.8633 Acc: 0.7089
# 
# Epoch 13/100
# ----------
# train Total Loss: 0.4652 Acc: 0.8575
# val Total Loss: 0.8504 Acc: 0.7131
# 
# Epoch 14/100
# ----------
# train Total Loss: 0.3953 Acc: 0.8841
# val Total Loss: 0.8620 Acc: 0.7300
# 
# Epoch 15/100
# ----------
# train Total Loss: 0.3714 Acc: 0.8953
# val Total Loss: 0.8713 Acc: 0.7089
# 
# Epoch 16/100
# ----------
# train Total Loss: 0.3154 Acc: 0.9134
# val Total Loss: 0.8746 Acc: 0.7089
# 
# Epoch 17/100
# ----------
# train Total Loss: 0.2885 Acc: 0.9246
# val Total Loss: 0.8716 Acc: 0.7173
# 
# Epoch 18/100
# ----------
# train Total Loss: 0.2828 Acc: 0.9064
# val Total Loss: 0.8958 Acc: 0.7004
# 
# Epoch 19/100
# ----------
# train Total Loss: 0.1929 Acc: 0.9553
# val Total Loss: 0.8492 Acc: 0.7215
# 
# Epoch 20/100
# ----------
# train Total Loss: 0.2053 Acc: 0.9539
# val Total Loss: 0.9147 Acc: 0.7215
# 
# Epoch 21/100
# ----------
# train Total Loss: 0.1914 Acc: 0.9427
# val Total Loss: 0.9050 Acc: 0.7215
# 
# Epoch 22/100
# ----------
# train Total Loss: 0.1592 Acc: 0.9581
# val Total Loss: 0.9391 Acc: 0.7131
# 
# Epoch 23/100
# ----------
# train Total Loss: 0.1636 Acc: 0.9595
# val Total Loss: 0.9991 Acc: 0.7004
# 
# Epoch 24/100
# ----------
# train Total Loss: 0.1554 Acc: 0.9637
# val Total Loss: 0.9903 Acc: 0.7131
# 
# Epoch 25/100
# ----------
# train Total Loss: 0.1262 Acc: 0.9763
# val Total Loss: 0.9599 Acc: 0.7089
# 
# Epoch 26/100
# ----------
# train Total Loss: 0.1245 Acc: 0.9777
# val Total Loss: 0.9398 Acc: 0.7089
# 
# Epoch 27/100
# ----------
# train Total Loss: 0.1210 Acc: 0.9735
# val Total Loss: 0.9750 Acc: 0.7004
# 
# Epoch 28/100
# ----------
# train Total Loss: 0.1088 Acc: 0.9735
# val Total Loss: 1.0203 Acc: 0.6962
# 
# Epoch 29/100
# ----------
# train Total Loss: 0.1209 Acc: 0.9679
# val Total Loss: 1.0455 Acc: 0.7131
# 
# Epoch 30/100
# ----------
# train Total Loss: 0.1083 Acc: 0.9735
# val Total Loss: 1.0152 Acc: 0.7257
# 
# Epoch 31/100
# ----------
# train Total Loss: 0.0702 Acc: 0.9888
# val Total Loss: 1.0135 Acc: 0.7384
# 
# Epoch 32/100
# ----------
# train Total Loss: 0.0744 Acc: 0.9846
# val Total Loss: 1.0114 Acc: 0.7300
# 
# Epoch 33/100
# ----------
# train Total Loss: 0.0714 Acc: 0.9818
# val Total Loss: 1.0067 Acc: 0.7215
# 
# Epoch 34/100
# ----------
# train Total Loss: 0.0907 Acc: 0.9818
# val Total Loss: 1.0575 Acc: 0.7173
# 
# Epoch 35/100
# ----------
# train Total Loss: 0.0801 Acc: 0.9818
# val Total Loss: 1.0158 Acc: 0.7173
# 
# Epoch 36/100
# ----------
# train Total Loss: 0.0705 Acc: 0.9874
# val Total Loss: 0.9990 Acc: 0.7215
# 
# Epoch 37/100
# ----------
# train Total Loss: 0.0956 Acc: 0.9679
# val Total Loss: 1.0183 Acc: 0.7300
# 
# Epoch 38/100
# ----------
# train Total Loss: 0.0850 Acc: 0.9846
# val Total Loss: 1.0155 Acc: 0.7215
# 
# Epoch 39/100
# ----------
# train Total Loss: 0.0870 Acc: 0.9860
# val Total Loss: 1.0132 Acc: 0.7257
# 
# Epoch 40/100
# ----------
# train Total Loss: 0.0727 Acc: 0.9832
# val Total Loss: 1.0232 Acc: 0.7300
# 
# Epoch 41/100
# ----------
# train Total Loss: 0.0726 Acc: 0.9804
# val Total Loss: 1.0424 Acc: 0.7257
# 
# Epoch 42/100
# ----------
# train Total Loss: 0.0828 Acc: 0.9804
# val Total Loss: 1.0155 Acc: 0.7215
# 
# Epoch 43/100
# ----------
# train Total Loss: 0.0702 Acc: 0.9763
# val Total Loss: 1.0226 Acc: 0.7300
# 
# Epoch 44/100
# ----------
# train Total Loss: 0.0657 Acc: 0.9846
# val Total Loss: 1.0022 Acc: 0.7342
# 
# Epoch 45/100
# ----------
# train Total Loss: 0.0822 Acc: 0.9846
# val Total Loss: 1.0125 Acc: 0.7215
# 
# Epoch 46/100
# ----------
# train Total Loss: 0.0734 Acc: 0.9860
# val Total Loss: 1.0158 Acc: 0.7257
# 
# Epoch 47/100
# ----------
# train Total Loss: 0.0652 Acc: 0.9888
# val Total Loss: 1.0143 Acc: 0.7384
# 
# Epoch 48/100
# ----------
# train Total Loss: 0.0621 Acc: 0.9860
# val Total Loss: 1.0294 Acc: 0.7215
# 
# Epoch 49/100
# ----------
# train Total Loss: 0.0873 Acc: 0.9818
# val Total Loss: 1.0605 Acc: 0.7300
# 
# Epoch 50/100
# ----------
# train Total Loss: 0.0726 Acc: 0.9874
# val Total Loss: 1.0124 Acc: 0.7300
# 
# Epoch 51/100
# ----------
# train Total Loss: 0.0760 Acc: 0.9791
# val Total Loss: 1.0356 Acc: 0.7089
# 
# Epoch 52/100
# ----------
# train Total Loss: 0.0695 Acc: 0.9791
# val Total Loss: 1.0329 Acc: 0.7089
# 
# Epoch 53/100
# ----------
# train Total Loss: 0.0821 Acc: 0.9777
# val Total Loss: 1.0320 Acc: 0.7300
# 
# Epoch 54/100
# ----------
# train Total Loss: 0.0740 Acc: 0.9846
# val Total Loss: 1.0209 Acc: 0.7215
# 
# Epoch 55/100
# ----------
# train Total Loss: 0.0725 Acc: 0.9846
# val Total Loss: 1.0052 Acc: 0.7300
# 
# Epoch 56/100
# ----------
# train Total Loss: 0.0679 Acc: 0.9846
# val Total Loss: 1.0141 Acc: 0.7300
# 
# Epoch 57/100
# ----------
# train Total Loss: 0.0523 Acc: 0.9916
# val Total Loss: 1.0325 Acc: 0.7046
# 
# Epoch 58/100
# ----------
# train Total Loss: 0.0715 Acc: 0.9832
# val Total Loss: 1.0344 Acc: 0.7131
# 
# Epoch 59/100
# ----------
# train Total Loss: 0.0669 Acc: 0.9860
# val Total Loss: 1.0125 Acc: 0.7384
# 
# Epoch 60/100
# ----------
# train Total Loss: 0.0637 Acc: 0.9888
# val Total Loss: 1.0079 Acc: 0.7300
# 
# Epoch 61/100
# ----------
# train Total Loss: 0.0623 Acc: 0.9846
# val Total Loss: 1.0313 Acc: 0.7131
# 
# Epoch 62/100
# ----------
# train Total Loss: 0.0825 Acc: 0.9749
# val Total Loss: 1.0084 Acc: 0.7089
# 
# Epoch 63/100
# ----------
# train Total Loss: 0.0638 Acc: 0.9846
# val Total Loss: 1.0469 Acc: 0.7342
# 
# Epoch 64/100
# ----------
# train Total Loss: 0.0867 Acc: 0.9763
# val Total Loss: 1.0330 Acc: 0.7215
# 
# Epoch 65/100
# ----------
# train Total Loss: 0.0668 Acc: 0.9846
# val Total Loss: 1.0222 Acc: 0.7173
# 
# Epoch 66/100
# ----------
# train Total Loss: 0.0924 Acc: 0.9749
# val Total Loss: 1.0309 Acc: 0.7300
# 
# Epoch 67/100
# ----------
# train Total Loss: 0.0753 Acc: 0.9777
# val Total Loss: 1.0141 Acc: 0.7257
# 
# Epoch 68/100
# ----------
# train Total Loss: 0.0599 Acc: 0.9902
# val Total Loss: 1.0508 Acc: 0.7173
# 
# Epoch 69/100
# ----------
# train Total Loss: 0.0678 Acc: 0.9832
# val Total Loss: 1.0393 Acc: 0.7215
# 
# Epoch 70/100
# ----------
# train Total Loss: 0.0769 Acc: 0.9791
# val Total Loss: 1.0066 Acc: 0.7257
# 
# Epoch 71/100
# ----------
# train Total Loss: 0.0923 Acc: 0.9749
# val Total Loss: 1.0176 Acc: 0.7257
# 
# Epoch 72/100
# ----------
# train Total Loss: 0.0511 Acc: 0.9888
# val Total Loss: 1.0274 Acc: 0.7300
# 
# Epoch 73/100
# ----------
# train Total Loss: 0.0837 Acc: 0.9791
# val Total Loss: 1.0457 Acc: 0.7131
# 
# Epoch 74/100
# ----------
# train Total Loss: 0.0736 Acc: 0.9846
# val Total Loss: 1.0089 Acc: 0.7257
# 
# Epoch 75/100
# ----------
# train Total Loss: 0.0852 Acc: 0.9860
# val Total Loss: 1.0165 Acc: 0.7257
# 
# Epoch 76/100
# ----------
# train Total Loss: 0.0806 Acc: 0.9818
# val Total Loss: 1.0389 Acc: 0.7131
# 
# Epoch 77/100
# ----------
# train Total Loss: 0.0659 Acc: 0.9888
# val Total Loss: 1.0404 Acc: 0.7004
# 
# Epoch 78/100
# ----------
# train Total Loss: 0.0669 Acc: 0.9846
# val Total Loss: 1.0308 Acc: 0.7173
# 
# Epoch 79/100
# ----------
# train Total Loss: 0.0726 Acc: 0.9804
# val Total Loss: 1.0151 Acc: 0.7215
# 
# Epoch 80/100
# ----------
# train Total Loss: 0.0696 Acc: 0.9888
# val Total Loss: 1.0366 Acc: 0.7215
# 
# Epoch 81/100
# ----------
# train Total Loss: 0.0669 Acc: 0.9902
# val Total Loss: 1.0176 Acc: 0.7257
# 
# Epoch 82/100
# ----------
# train Total Loss: 0.0758 Acc: 0.9832
# val Total Loss: 1.0217 Acc: 0.7300
# 
# Epoch 83/100
# ----------
# train Total Loss: 0.0574 Acc: 0.9902
# val Total Loss: 1.0269 Acc: 0.7300
# 
# Epoch 84/100
# ----------
# train Total Loss: 0.0630 Acc: 0.9846
# val Total Loss: 1.0272 Acc: 0.7300
# 
# Epoch 85/100
# ----------
# train Total Loss: 0.0598 Acc: 0.9874
# val Total Loss: 1.0240 Acc: 0.7257
# 
# Epoch 86/100
# ----------
# train Total Loss: 0.0696 Acc: 0.9804
# val Total Loss: 1.0150 Acc: 0.7215
# 
# Epoch 87/100
# ----------
# train Total Loss: 0.0869 Acc: 0.9804
# val Total Loss: 1.0363 Acc: 0.7131
# 
# Epoch 88/100
# ----------
# train Total Loss: 0.0732 Acc: 0.9846
# val Total Loss: 1.0099 Acc: 0.7215
# 
# Epoch 89/100
# ----------
# train Total Loss: 0.0603 Acc: 0.9874
# val Total Loss: 1.0187 Acc: 0.7215
# 
# Epoch 90/100
# ----------
# train Total Loss: 0.0752 Acc: 0.9832
# val Total Loss: 1.0523 Acc: 0.7173
# 
# Epoch 91/100
# ----------
# train Total Loss: 0.0642 Acc: 0.9888
# val Total Loss: 1.0226 Acc: 0.7257
# 
# Epoch 92/100
# ----------
# train Total Loss: 0.0757 Acc: 0.9832
# val Total Loss: 1.0493 Acc: 0.7257
# 
# Epoch 93/100
# ----------
# train Total Loss: 0.0636 Acc: 0.9874
# val Total Loss: 1.0130 Acc: 0.7173
# 
# Epoch 94/100
# ----------
# train Total Loss: 0.0552 Acc: 0.9916
# val Total Loss: 1.0290 Acc: 0.7131
# 
# Epoch 95/100
# ----------
# train Total Loss: 0.0728 Acc: 0.9832
# val Total Loss: 1.0106 Acc: 0.7342
# 
# Epoch 96/100
# ----------
# train Total Loss: 0.0916 Acc: 0.9749
# val Total Loss: 1.0514 Acc: 0.7173
# 
# Epoch 97/100
# ----------
# train Total Loss: 0.0791 Acc: 0.9791
# val Total Loss: 1.0133 Acc: 0.7173
# 
# Epoch 98/100
# ----------
# train Total Loss: 0.0638 Acc: 0.9888
# val Total Loss: 0.9987 Acc: 0.7173
# 
# Epoch 99/100
# ----------
# train Total Loss: 0.0831 Acc: 0.9846
# val Total Loss: 0.9983 Acc: 0.7257
# 
# Epoch 100/100
# ----------
# train Total Loss: 0.0598 Acc: 0.9888
# val Total Loss: 1.0248 Acc: 0.7257
# 
# Best val Acc: 0.7384
#
# %% [markdown]
# 
# -> ?ш쾶 ?뺥솗?꾧? ?μ긽?섏????딆? 寃껋쓣 ?뺤씤 (epoch媛 ?꾨땶 ?ㅻⅨ ?붿냼瑜?蹂寃쏀븯??ex.?숈뒿 ?곗씠?곗뼇 利앷?) ?뺥솗?꾨? ?щ젮?쇳븷 寃껋씠?쇨퀬 ?덉륫??
#

