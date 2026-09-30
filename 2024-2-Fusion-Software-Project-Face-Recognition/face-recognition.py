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
# Requirement already satisfied: split-folders in /usr/local/lib/python3.10/dist-packages (0.5.1)
# 
#
# %%
!pip install deepface
# %% [markdown]
# Saved output
# Requirement already satisfied: deepface in /usr/local/lib/python3.10/dist-packages (0.0.93)
# Requirement already satisfied: requests>=2.27.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.32.3)
# Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.26.4)
# Requirement already satisfied: pandas>=0.23.4 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.2.2)
# Requirement already satisfied: gdown>=3.10.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (5.2.0)
# Requirement already satisfied: tqdm>=4.30.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.67.1)
# Requirement already satisfied: Pillow>=5.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (9.0.1)
# Requirement already satisfied: opencv-python>=4.5.5.64 in /usr/local/lib/python3.10/dist-packages (from deepface) (4.10.0.84)
# Requirement already satisfied: tensorflow>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (2.17.1)
# Requirement already satisfied: keras>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (3.5.0)
# Requirement already satisfied: Flask>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from deepface) (3.1.0)
# Requirement already satisfied: flask-cors>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (5.0.0)
# Requirement already satisfied: mtcnn>=0.1.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (1.0.0)
# Requirement already satisfied: retina-face>=0.0.1 in /usr/local/lib/python3.10/dist-packages (from deepface) (0.0.17)
# Requirement already satisfied: fire>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (0.7.0)
# Requirement already satisfied: gunicorn>=20.1.0 in /usr/local/lib/python3.10/dist-packages (from deepface) (23.0.0)
# Requirement already satisfied: termcolor in /usr/local/lib/python3.10/dist-packages (from fire>=0.4.0->deepface) (2.5.0)
# Requirement already satisfied: Werkzeug>=3.1 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.1.3)
# Requirement already satisfied: Jinja2>=3.1.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (3.1.4)
# Requirement already satisfied: itsdangerous>=2.2 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (2.2.0)
# Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (8.1.7)
# Requirement already satisfied: blinker>=1.9 in /usr/local/lib/python3.10/dist-packages (from Flask>=1.1.2->deepface) (1.9.0)
# Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (4.12.3)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from gdown>=3.10.1->deepface) (3.16.1)
# Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gunicorn>=20.1.0->deepface) (24.2)
# Requirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (1.4.0)
# Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (13.9.4)
# Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.0.8)
# Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (3.12.1)
# Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.13.1)
# Requirement already satisfied: ml-dtypes in /usr/local/lib/python3.10/dist-packages (from keras>=2.2.0->deepface) (0.4.1)
# Requirement already satisfied: joblib>=1.4.2 in /usr/local/lib/python3.10/dist-packages (from mtcnn>=0.1.0->deepface) (1.4.2)
# Requirement already satisfied: lz4>=4.3.3 in /usr/local/lib/python3.10/dist-packages (from mtcnn>=0.1.0->deepface) (4.3.3)
# Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2.8.2)
# Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2024.2)
# Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=0.23.4->deepface) (2024.2)
# Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (3.4.0)
# Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (3.10)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (2.2.3)
# Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.27.1->deepface) (2024.12.14)
# Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.6.3)
# Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (24.3.25)
# Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.6.0)
# Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.2.0)
# Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (18.1.1)
# Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (3.4.0)
# Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (4.25.5)
# Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (75.1.0)
# Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.17.0)
# Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (4.12.2)
# Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.17.0)
# Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (1.68.1)
# Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (2.17.1)
# Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow>=1.9.0->deepface) (0.37.1)
# Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow>=1.9.0->deepface) (0.45.1)
# Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from Jinja2>=3.1.2->Flask>=1.1.2->deepface) (3.0.2)
# Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow>=1.9.0->deepface) (3.7)
# Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow>=1.9.0->deepface) (0.7.2)
# Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->gdown>=3.10.1->deepface) (2.6)
# Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /usr/local/lib/python3.10/dist-packages (from requests[socks]->gdown>=3.10.1->deepface) (1.7.1)
# Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=2.2.0->deepface) (3.0.0)
# Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=2.2.0->deepface) (2.18.0)
# Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=2.2.0->deepface) (0.1.2)
# 
#
# %%
!pip install efficientnet-pytorch
# %% [markdown]
# Saved output
# Requirement already satisfied: efficientnet-pytorch in /usr/local/lib/python3.10/dist-packages (0.7.1)
# Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from efficientnet-pytorch) (2.5.1+cu121)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (3.16.1)
# Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (4.12.2)
# Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (3.4.2)
# Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (3.1.4)
# Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (2024.10.0)
# Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch->efficientnet-pytorch) (1.13.1)
# Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch->efficientnet-pytorch) (1.3.0)
# Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->efficientnet-pytorch) (3.0.2)
# 
#
# %%
!pip install vggface
# %% [markdown]
# Saved output
# Requirement already satisfied: vggface in /usr/local/lib/python3.10/dist-packages (1.0.0)
# Requirement already satisfied: pillow==9.0.1 in /usr/local/lib/python3.10/dist-packages (from vggface) (9.0.1)
# 
#
# %%
!pip install torchvision
# %% [markdown]
# Saved output
# Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.20.1+cu121)
# Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.26.4)
# Requirement already satisfied: torch==2.5.1 in /usr/local/lib/python3.10/dist-packages (from torchvision) (2.5.1+cu121)
# Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (11.0.0)
# Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (3.16.1)
# Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (4.12.2)
# Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (3.4.2)
# Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (3.1.4)
# Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (2024.10.0)
# Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch==2.5.1->torchvision) (1.13.1)
# Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch==2.5.1->torchvision) (1.3.0)
# Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch==2.5.1->torchvision) (3.0.2)
# 
#
# %%
from torchvision.models import mobilenet_v2
# %%
from vggface import VGGFace
from deepface import DeepFace
from efficientnet_pytorch import EfficientNet
from torchvision.models import mobilenet_v2
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

train_zip_path = '/content/drive/MyDrive/2024-2 ?뗡뀼?솽꼺?■넾?됣뀳?묃뀽?먤뀽?뗡뀺?뗡뀯?묃뀽?끷뀳?뚡뀰?ⓤ꼸??train.zip'
test_zip_path = '/content/drive/MyDrive/2024-2 ?뗡뀼?솽꼺?■넾?됣뀳?묃뀽?먤뀽?뗡뀺?뗡뀯?묃뀽?끷뀳?뚡뀰?ⓤ꼸??test.zip'

# ?뺤텞 ?댁젣
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/train')
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/test')
# %%
import os
import shutil

# 湲곗〈 train ?대뜑 寃쎈줈
train_dir = '/content/train'

# train ?대뜑 ?댁쓽 遺덊븘?뷀븳 'train' ?대뜑 ?쒓굅
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path) and class_name == 'train':
        shutil.rmtree(class_path)  # 'train' ?대뜑 ??젣

# 媛??대옒???대뜑 ?댁뿉 ?대?吏 ?뚯씪?ㅼ씠 ?덈뒗吏 ?뺤씤
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        print(f"Class folder: {class_name}, Files: {os.listdir(class_path)}")
# %% [markdown]
# Saved output
# Class folder: kangdaniel, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '1145.jpg', '768.jpg', '930.jpg', '1119.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '1158.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '1089.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '1247.jpg', '939.jpg', '202.jpg', '283.jpg', '1175.jpg', '1194.jpg', '14.jpg', '102.jpg', '1243.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1095.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1179.jpg', '468.jpg', '229.jpg', '132.jpg', '1210.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '1236.jpg', '1154.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '1104.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '1101.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '1193.jpg', '638.jpg', '954.jpg', '282.jpg', '1244.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '1240.jpg', '134.jpg', '346.jpg', '731.jpg', '1083.jpg', '50.jpg', '914.jpg', '1092.jpg', '378.jpg', '1184.jpg', '812.jpg', '1205.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '1252.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '1141.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '891.jpg', '441.jpg', '189.jpg', '1105.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '1146.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '1093.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '1087.jpg', '1229.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '1165.jpg', '225.jpg', '142.jpg', '1233.jpg', '405.jpg', '1206.jpg', '1181.jpg', '1218.jpg', '273.jpg', '1201.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '1144.jpg', '119.jpg', '191.jpg', '107.jpg', '72.jpg', '170.jpg', '1251.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '1110.jpg', '1207.jpg', '838.jpg', '888.jpg', '1094.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '1149.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '1186.jpg', '1147.jpg', '1162.jpg', '1082.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '1204.jpg', '1117.jpg', '113.jpg', '389.jpg', '1075.jpg', '618.jpg', '857.jpg', '1150.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '407.jpg', '388.jpg', '1220.jpg', '511.jpg', '92.jpg', '1138.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '1249.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '1159.jpg', '1128.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '1153.jpg', '1187.jpg', '669.jpg', '199.jpg', '1134.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '1122.jpg', '1191.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '1160.jpg', '1109.jpg', '1148.jpg', '1080.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1157.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '1170.jpg', '1198.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '1163.jpg', '1106.jpg', '782.jpg', '777.jpg', '1192.jpg', '663.jpg', '923.jpg', '595.jpg', '1169.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '1221.jpg', '769.jpg', '446.jpg', '11.jpg', '1114.jpg', '1177.jpg', '754.jpg', '204.jpg', '1142.jpg', '43.jpg', '713.jpg', '1235.jpg', '110.jpg', '1084.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '1098.jpg', '212.jpg', '591.jpg', '906.jpg', '1173.jpg', '541.jpg', '71.jpg', '1231.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '1241.jpg', '885.jpg', '352.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '515.jpg', '324.jpg', '1099.jpg', '1077.jpg', '734.jpg', '1108.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '1225.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '1115.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '1140.jpg', '99.jpg', '547.jpg', '702.jpg', '1203.jpg', '193.jpg', '904.jpg', '856.jpg', '1178.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '291.jpg', '65.jpg', '264.jpg', '1086.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '5.jpg', '1137.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '1076.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '1212.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '1102.jpg', '416.jpg', '492.jpg', '1199.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '1239.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '1180.jpg', '54.jpg', '1096.jpg', '1213.jpg', '1237.jpg', '560.jpg', '1079.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '1164.jpg', '644.jpg', '1152.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '1088.jpg', '1085.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '1185.jpg', '442.jpg', '86.jpg', '1074.jpg', '1124.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '1127.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '1242.jpg', '1254.jpg', '403.jpg', '1214.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '1248.jpg', '1250.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '82.jpg', '933.jpg', '1072.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '1232.jpg', '514.jpg', '242.jpg', '1224.jpg', '28.jpg', '1166.jpg', '461.jpg', '94.jpg', '1238.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '1172.jpg', '650.jpg', '1196.jpg', '750.jpg', '825.jpg', '1103.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '1176.jpg', '323.jpg', '1126.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '1151.jpg', '536.jpg', '775.jpg', '1202.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '1139.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '1125.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '1208.jpg', '946.jpg', '139.jpg', '1155.jpg', '233.jpg', '1116.jpg', '256.jpg', '1118.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '1215.jpg', '1078.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '1219.jpg', '943.jpg', '548.jpg', '481.jpg', '1107.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '1200.jpg', '771.jpg', '1113.jpg', '1226.jpg', '125.jpg', '928.jpg', '275.jpg', '1195.jpg', '497.jpg', '660.jpg', '1136.jpg', '549.jpg', '1156.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '1188.jpg', '1091.jpg', '724.jpg', '485.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '1135.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '1209.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '1132.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '1167.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '1253.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '1217.jpg', '1073.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '1097.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '1168.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '1123.jpg', '842.jpg', '625.jpg', '873.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '1245.jpg', '1182.jpg', '1246.jpg', '1197.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '1216.jpg', '21.jpg', '396.jpg', '369.jpg', '1228.jpg', '309.jpg', '1183.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '1120.jpg', '1090.jpg', '31.jpg', '61.jpg', '1133.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '1171.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '1121.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1234.jpg', '1112.jpg', '297.jpg', '1189.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '1190.jpg', '355.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '712.jpg', '410.jpg', '1211.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '1174.jpg', '799.jpg', '210.jpg', '1130.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '1081.jpg', '391.jpg', '129.jpg', '166.jpg', '1111.jpg', '586.jpg', '819.jpg', '539.jpg', '1129.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '1227.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '1230.jpg', '1131.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '1223.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '1100.jpg', '1222.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '1143.jpg', '1161.jpg', '530.jpg']
# Class folder: skz_hunjin, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1020.jpg', '468.jpg', '1005.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '1011.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '1022.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '1003.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '1021.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1023.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1012.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '1019.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '1015.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '1018.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '1002.jpg', '560.jpg', '1014.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '1007.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '1016.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '1009.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '1017.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '1010.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '1008.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '1004.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1013.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '1006.jpg', '671.jpg', '1024.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: bts_v, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '468.jpg', '1005.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '1011.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '1003.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1012.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '1015.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '1002.jpg', '560.jpg', '1014.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '1007.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '1016.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '1009.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '1010.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '1008.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '1004.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1013.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '1006.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: Kimyuna, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '468.jpg', '1005.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '1011.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '1003.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1012.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '1002.jpg', '560.jpg', '1014.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '1007.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '1009.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '1010.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '1008.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '1004.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1013.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '1006.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: ive_jangwonyoung, Files: ['821.jpg', '440.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '1145.jpg', '768.jpg', '1119.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '1158.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '1089.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '1247.jpg', '202.jpg', '283.jpg', '1175.jpg', '1194.jpg', '14.jpg', '1256.jpg', '102.jpg', '1243.jpg', '463.jpg', '1063.jpg', '223.jpg', '892.jpg', '805.jpg', '1095.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1179.jpg', '468.jpg', '229.jpg', '132.jpg', '1210.jpg', '1038.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '1236.jpg', '1154.jpg', '371.jpg', '334.jpg', '296.jpg', '1027.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '49.jpg', '710.jpg', '1069.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '1104.jpg', '1052.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '1101.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '1193.jpg', '638.jpg', '282.jpg', '1244.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '1068.jpg', '447.jpg', '882.jpg', '1240.jpg', '134.jpg', '346.jpg', '731.jpg', '1083.jpg', '50.jpg', '914.jpg', '1092.jpg', '378.jpg', '1184.jpg', '812.jpg', '1205.jpg', '604.jpg', '633.jpg', '1252.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '1141.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '1048.jpg', '570.jpg', '483.jpg', '400.jpg', '891.jpg', '441.jpg', '189.jpg', '1105.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '1146.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '1093.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '1043.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '1037.jpg', '603.jpg', '18.jpg', '375.jpg', '322.jpg', '1087.jpg', '1229.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '1165.jpg', '225.jpg', '142.jpg', '1233.jpg', '405.jpg', '1034.jpg', '1206.jpg', '1181.jpg', '1218.jpg', '273.jpg', '1201.jpg', '415.jpg', '1039.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '1144.jpg', '119.jpg', '191.jpg', '107.jpg', '72.jpg', '170.jpg', '1251.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '1110.jpg', '1207.jpg', '838.jpg', '888.jpg', '1094.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '1149.jpg', '828.jpg', '462.jpg', '1.jpg', '36.jpg', '175.jpg', '1186.jpg', '1147.jpg', '1162.jpg', '1082.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '1204.jpg', '1117.jpg', '113.jpg', '389.jpg', '1075.jpg', '618.jpg', '857.jpg', '1150.jpg', '834.jpg', '390.jpg', '627.jpg', '407.jpg', '388.jpg', '1059.jpg', '1220.jpg', '511.jpg', '92.jpg', '1138.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '295.jpg', '443.jpg', '1070.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '1249.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '1159.jpg', '1128.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '1153.jpg', '1187.jpg', '669.jpg', '199.jpg', '1134.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '1122.jpg', '1191.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '1160.jpg', '1109.jpg', '1148.jpg', '1080.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1157.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '781.jpg', '519.jpg', '70.jpg', '218.jpg', '380.jpg', '226.jpg', '705.jpg', '244.jpg', '729.jpg', '1258.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '1170.jpg', '1198.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '1163.jpg', '1106.jpg', '782.jpg', '1025.jpg', '777.jpg', '1192.jpg', '663.jpg', '923.jpg', '595.jpg', '1169.jpg', '471.jpg', '1058.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '1221.jpg', '769.jpg', '446.jpg', '11.jpg', '1031.jpg', '1114.jpg', '1177.jpg', '754.jpg', '204.jpg', '1142.jpg', '43.jpg', '713.jpg', '1235.jpg', '110.jpg', '1084.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '1098.jpg', '212.jpg', '591.jpg', '906.jpg', '1173.jpg', '541.jpg', '71.jpg', '1231.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '1055.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '1241.jpg', '885.jpg', '352.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '1067.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1260.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '515.jpg', '1035.jpg', '324.jpg', '1099.jpg', '1077.jpg', '734.jpg', '1029.jpg', '1108.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '1225.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '1115.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '1140.jpg', '99.jpg', '547.jpg', '702.jpg', '1203.jpg', '193.jpg', '904.jpg', '1049.jpg', '856.jpg', '1178.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '291.jpg', '65.jpg', '264.jpg', '1086.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '5.jpg', '1137.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '1076.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '1050.jpg', '427.jpg', '1212.jpg', '239.jpg', '1259.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '1102.jpg', '416.jpg', '492.jpg', '1199.jpg', '284.jpg', '1060.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '1239.jpg', '147.jpg', '1045.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '1180.jpg', '54.jpg', '1096.jpg', '1213.jpg', '1032.jpg', '1237.jpg', '560.jpg', '1079.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '599.jpg', '417.jpg', '419.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '1026.jpg', '1164.jpg', '644.jpg', '1152.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '1040.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '1088.jpg', '1085.jpg', '340.jpg', '681.jpg', '332.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '1185.jpg', '442.jpg', '86.jpg', '1074.jpg', '1124.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '639.jpg', '238.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '1127.jpg', '875.jpg', '1257.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '1242.jpg', '1254.jpg', '403.jpg', '1214.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '1248.jpg', '1250.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '82.jpg', '1054.jpg', '1072.jpg', '161.jpg', '1033.jpg', '791.jpg', '421.jpg', '105.jpg', '1232.jpg', '514.jpg', '242.jpg', '1224.jpg', '28.jpg', '1166.jpg', '461.jpg', '94.jpg', '1238.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '1172.jpg', '650.jpg', '1196.jpg', '750.jpg', '825.jpg', '1103.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '1176.jpg', '323.jpg', '1126.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '1151.jpg', '536.jpg', '775.jpg', '1202.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '1139.jpg', '475.jpg', '592.jpg', '1062.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '1125.jpg', '13.jpg', '1030.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '1208.jpg', '139.jpg', '1155.jpg', '233.jpg', '1116.jpg', '1065.jpg', '256.jpg', '1118.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '1215.jpg', '1078.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '1061.jpg', '1056.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '1219.jpg', '548.jpg', '481.jpg', '1107.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '1200.jpg', '771.jpg', '1113.jpg', '1226.jpg', '125.jpg', '275.jpg', '1195.jpg', '497.jpg', '660.jpg', '1136.jpg', '549.jpg', '1156.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '537.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '1041.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '231.jpg', '1188.jpg', '1091.jpg', '724.jpg', '485.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '1135.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '310.jpg', '1028.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '877.jpg', '91.jpg', '196.jpg', '1255.jpg', '698.jpg', '569.jpg', '208.jpg', '1209.jpg', '790.jpg', '818.jpg', '1053.jpg', '128.jpg', '865.jpg', '1132.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '1167.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '1253.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '1046.jpg', '1217.jpg', '1073.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '1064.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '206.jpg', '409.jpg', '789.jpg', '1047.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '1097.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '1168.jpg', '435.jpg', '148.jpg', '657.jpg', '377.jpg', '814.jpg', '1057.jpg', '455.jpg', '1123.jpg', '842.jpg', '625.jpg', '873.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '859.jpg', '808.jpg', '1245.jpg', '1182.jpg', '1246.jpg', '1197.jpg', '85.jpg', '16.jpg', '1051.jpg', '716.jpg', '351.jpg', '711.jpg', '1216.jpg', '21.jpg', '396.jpg', '369.jpg', '1228.jpg', '309.jpg', '1183.jpg', '6.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '97.jpg', '543.jpg', '1042.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '1120.jpg', '1090.jpg', '31.jpg', '61.jpg', '1133.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '1171.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '1121.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1234.jpg', '1112.jpg', '297.jpg', '1189.jpg', '596.jpg', '194.jpg', '869.jpg', '1190.jpg', '355.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '712.jpg', '410.jpg', '1211.jpg', '751.jpg', '668.jpg', '160.jpg', '678.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '1174.jpg', '799.jpg', '210.jpg', '1130.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '1071.jpg', '753.jpg', '360.jpg', '169.jpg', '1036.jpg', '1081.jpg', '391.jpg', '129.jpg', '166.jpg', '1111.jpg', '586.jpg', '819.jpg', '539.jpg', '1044.jpg', '1129.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '1066.jpg', '778.jpg', '855.jpg', '1227.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '1230.jpg', '1131.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '671.jpg', '1024.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '1223.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '1100.jpg', '1222.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '1143.jpg', '1161.jpg', '530.jpg']
# Class folder: IU, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1020.jpg', '468.jpg', '1005.jpg', '229.jpg', '132.jpg', '1038.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '1027.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '1011.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '1052.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '1022.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '1048.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '1043.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '1037.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '1034.jpg', '273.jpg', '415.jpg', '1039.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '1003.jpg', '407.jpg', '388.jpg', '1059.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '1021.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1023.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '1025.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '1058.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '1031.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '1055.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1012.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '1035.jpg', '324.jpg', '734.jpg', '1029.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '1049.jpg', '856.jpg', '1019.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '1015.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '1018.jpg', '800.jpg', '583.jpg', '1050.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '1060.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '1045.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '1002.jpg', '1032.jpg', '560.jpg', '1014.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '1026.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '1040.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '1007.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '1016.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '1009.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '1054.jpg', '933.jpg', '161.jpg', '1033.jpg', '791.jpg', '421.jpg', '105.jpg', '1017.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '1030.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '1056.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '1041.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '1010.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '1028.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '1053.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '1046.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '1047.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '1057.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '1008.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '1051.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '1042.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '1004.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1013.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '1036.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '1044.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '1006.jpg', '671.jpg', '1024.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: bts_jimin, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '1145.jpg', '768.jpg', '930.jpg', '1119.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '1158.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '1175.jpg', '1194.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1179.jpg', '468.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '1154.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '1104.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '1193.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '1184.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '1141.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '1105.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '1146.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '1165.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '1181.jpg', '273.jpg', '1201.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '1144.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '1110.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '1149.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '1186.jpg', '1147.jpg', '1162.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '1204.jpg', '1117.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '1150.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '1138.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '1159.jpg', '1128.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '1153.jpg', '1187.jpg', '669.jpg', '199.jpg', '1134.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '1122.jpg', '1191.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '1160.jpg', '1109.jpg', '1148.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1157.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '1170.jpg', '1198.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '1163.jpg', '1106.jpg', '782.jpg', '777.jpg', '983.jpg', '1192.jpg', '663.jpg', '923.jpg', '595.jpg', '1169.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '1114.jpg', '1177.jpg', '754.jpg', '204.jpg', '1142.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '1173.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '1108.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '1115.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '1140.jpg', '99.jpg', '547.jpg', '702.jpg', '1203.jpg', '193.jpg', '904.jpg', '856.jpg', '1178.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '1137.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '1199.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '1180.jpg', '54.jpg', '1002.jpg', '560.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '1164.jpg', '644.jpg', '1152.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '1185.jpg', '442.jpg', '86.jpg', '1124.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '1127.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '514.jpg', '242.jpg', '28.jpg', '1166.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '1172.jpg', '650.jpg', '1196.jpg', '750.jpg', '825.jpg', '1103.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '1176.jpg', '323.jpg', '1126.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '1151.jpg', '536.jpg', '775.jpg', '1202.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '1139.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '1125.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '1155.jpg', '233.jpg', '1116.jpg', '256.jpg', '1118.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '1107.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '1200.jpg', '771.jpg', '1113.jpg', '125.jpg', '928.jpg', '275.jpg', '1195.jpg', '497.jpg', '660.jpg', '1136.jpg', '549.jpg', '1156.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '1188.jpg', '724.jpg', '485.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '1135.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '1132.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '1167.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '1168.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '1123.jpg', '842.jpg', '625.jpg', '873.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '1182.jpg', '1197.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '1183.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '1120.jpg', '981.jpg', '31.jpg', '61.jpg', '1133.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '1171.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '1121.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1112.jpg', '297.jpg', '1189.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '1190.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '1174.jpg', '799.jpg', '210.jpg', '1130.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '1111.jpg', '586.jpg', '819.jpg', '539.jpg', '1129.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '1131.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '1143.jpg', '1161.jpg', '530.jpg']
# Class folder: astro_chaeunwoo, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '1020.jpg', '468.jpg', '1005.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '1027.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '1011.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '1022.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '1003.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '1021.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '1023.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '1025.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '1012.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '1019.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '1015.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '1018.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '1002.jpg', '560.jpg', '1014.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '1026.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '1007.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '1016.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '1009.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '1017.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '1010.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '1028.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '1008.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '1004.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '1013.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '1001.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '1006.jpg', '671.jpg', '1024.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: aespa_karina, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '1000.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '468.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '560.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# Class folder: rv_irene, Files: ['821.jpg', '440.jpg', '938.jpg', '315.jpg', '682.jpg', '600.jpg', '878.jpg', '768.jpg', '930.jpg', '456.jpg', '279.jpg', '464.jpg', '670.jpg', '336.jpg', '385.jpg', '845.jpg', '582.jpg', '444.jpg', '564.jpg', '257.jpg', '230.jpg', '760.jpg', '666.jpg', '939.jpg', '202.jpg', '283.jpg', '14.jpg', '102.jpg', '463.jpg', '223.jpg', '892.jpg', '805.jpg', '557.jpg', '770.jpg', '458.jpg', '795.jpg', '505.jpg', '237.jpg', '249.jpg', '468.jpg', '229.jpg', '132.jpg', '4.jpg', '112.jpg', '145.jpg', '866.jpg', '934.jpg', '115.jpg', '162.jpg', '359.jpg', '277.jpg', '371.jpg', '334.jpg', '296.jpg', '798.jpg', '372.jpg', '749.jpg', '527.jpg', '886.jpg', '506.jpg', '676.jpg', '597.jpg', '969.jpg', '30.jpg', '822.jpg', '478.jpg', '767.jpg', '413.jpg', '303.jpg', '136.jpg', '642.jpg', '448.jpg', '350.jpg', '903.jpg', '49.jpg', '710.jpg', '703.jpg', '198.jpg', '686.jpg', '265.jpg', '520.jpg', '130.jpg', '290.jpg', '606.jpg', '747.jpg', '579.jpg', '213.jpg', '153.jpg', '707.jpg', '723.jpg', '367.jpg', '665.jpg', '195.jpg', '638.jpg', '954.jpg', '995.jpg', '282.jpg', '361.jpg', '431.jpg', '487.jpg', '516.jpg', '40.jpg', '958.jpg', '893.jpg', '454.jpg', '356.jpg', '725.jpg', '321.jpg', '955.jpg', '897.jpg', '843.jpg', '532.jpg', '262.jpg', '447.jpg', '882.jpg', '134.jpg', '346.jpg', '731.jpg', '50.jpg', '914.jpg', '378.jpg', '812.jpg', '957.jpg', '604.jpg', '633.jpg', '931.jpg', '60.jpg', '348.jpg', '699.jpg', '883.jpg', '433.jpg', '715.jpg', '127.jpg', '538.jpg', '12.jpg', '664.jpg', '15.jpg', '742.jpg', '287.jpg', '27.jpg', '570.jpg', '483.jpg', '400.jpg', '996.jpg', '891.jpg', '441.jpg', '189.jpg', '921.jpg', '156.jpg', '853.jpg', '319.jpg', '846.jpg', '975.jpg', '10.jpg', '23.jpg', '759.jpg', '178.jpg', '765.jpg', '114.jpg', '881.jpg', '876.jpg', '22.jpg', '626.jpg', '39.jpg', '572.jpg', '803.jpg', '301.jpg', '827.jpg', '89.jpg', '911.jpg', '776.jpg', '521.jpg', '90.jpg', '522.jpg', '370.jpg', '677.jpg', '422.jpg', '797.jpg', '603.jpg', '962.jpg', '18.jpg', '375.jpg', '322.jpg', '459.jpg', '894.jpg', '643.jpg', '289.jpg', '584.jpg', '222.jpg', '225.jpg', '142.jpg', '980.jpg', '405.jpg', '273.jpg', '415.jpg', '937.jpg', '809.jpg', '58.jpg', '453.jpg', '312.jpg', '119.jpg', '191.jpg', '107.jpg', '977.jpg', '72.jpg', '170.jpg', '457.jpg', '19.jpg', '811.jpg', '254.jpg', '594.jpg', '529.jpg', '577.jpg', '183.jpg', '542.jpg', '51.jpg', '813.jpg', '75.jpg', '978.jpg', '838.jpg', '888.jpg', '589.jpg', '804.jpg', '833.jpg', '123.jpg', '432.jpg', '188.jpg', '387.jpg', '829.jpg', '755.jpg', '828.jpg', '462.jpg', '960.jpg', '1.jpg', '36.jpg', '175.jpg', '259.jpg', '863.jpg', '890.jpg', '493.jpg', '517.jpg', '113.jpg', '389.jpg', '618.jpg', '857.jpg', '834.jpg', '390.jpg', '627.jpg', '942.jpg', '407.jpg', '388.jpg', '511.jpg', '92.jpg', '305.jpg', '227.jpg', '509.jpg', '35.jpg', '966.jpg', '414.jpg', '864.jpg', '314.jpg', '524.jpg', '989.jpg', '621.jpg', '667.jpg', '788.jpg', '785.jpg', '948.jpg', '295.jpg', '443.jpg', '998.jpg', '610.jpg', '399.jpg', '685.jpg', '672.jpg', '967.jpg', '985.jpg', '488.jpg', '260.jpg', '224.jpg', '915.jpg', '234.jpg', '245.jpg', '763.jpg', '42.jpg', '854.jpg', '53.jpg', '182.jpg', '44.jpg', '33.jpg', '722.jpg', '525.jpg', '46.jpg', '381.jpg', '641.jpg', '250.jpg', '251.jpg', '669.jpg', '199.jpg', '267.jpg', '167.jpg', '840.jpg', '420.jpg', '0.jpg', '552.jpg', '425.jpg', '235.jpg', '269.jpg', '3.jpg', '205.jpg', '697.jpg', '762.jpg', '266.jpg', '154.jpg', '327.jpg', '285.jpg', '299.jpg', '317.jpg', '656.jpg', '629.jpg', '180.jpg', '634.jpg', '761.jpg', '55.jpg', '924.jpg', '781.jpg', '519.jpg', '70.jpg', '925.jpg', '218.jpg', '380.jpg', '226.jpg', '970.jpg', '705.jpg', '244.jpg', '729.jpg', '24.jpg', '133.jpg', '898.jpg', '558.jpg', '766.jpg', '373.jpg', '384.jpg', '658.jpg', '430.jpg', '700.jpg', '782.jpg', '777.jpg', '983.jpg', '663.jpg', '923.jpg', '595.jpg', '471.jpg', '706.jpg', '653.jpg', '616.jpg', '645.jpg', '163.jpg', '769.jpg', '446.jpg', '11.jpg', '988.jpg', '754.jpg', '204.jpg', '43.jpg', '713.jpg', '110.jpg', '354.jpg', '756.jpg', '243.jpg', '382.jpg', '186.jpg', '402.jpg', '88.jpg', '423.jpg', '363.jpg', '212.jpg', '591.jpg', '906.jpg', '541.jpg', '71.jpg', '945.jpg', '683.jpg', '837.jpg', '733.jpg', '861.jpg', '602.jpg', '490.jpg', '83.jpg', '174.jpg', '179.jpg', '486.jpg', '34.jpg', '302.jpg', '173.jpg', '885.jpg', '352.jpg', '994.jpg', '708.jpg', '796.jpg', '787.jpg', '561.jpg', '871.jpg', '326.jpg', '860.jpg', '465.jpg', '607.jpg', '576.jpg', '736.jpg', '190.jpg', '540.jpg', '117.jpg', '508.jpg', '867.jpg', '406.jpg', '331.jpg', '990.jpg', '515.jpg', '324.jpg', '734.jpg', '80.jpg', '395.jpg', '428.jpg', '578.jpg', '159.jpg', '68.jpg', '313.jpg', '474.jpg', '810.jpg', '476.jpg', '298.jpg', '841.jpg', '126.jpg', '460.jpg', '801.jpg', '176.jpg', '631.jpg', '737.jpg', '720.jpg', '342.jpg', '99.jpg', '547.jpg', '702.jpg', '193.jpg', '904.jpg', '856.jpg', '598.jpg', '106.jpg', '913.jpg', '66.jpg', '318.jpg', '268.jpg', '984.jpg', '291.jpg', '65.jpg', '264.jpg', '271.jpg', '62.jpg', '501.jpg', '489.jpg', '918.jpg', '987.jpg', '5.jpg', '41.jpg', '651.jpg', '908.jpg', '601.jpg', '64.jpg', '684.jpg', '546.jpg', '800.jpg', '583.jpg', '427.jpg', '971.jpg', '239.jpg', '534.jpg', '241.jpg', '164.jpg', '286.jpg', '503.jpg', '944.jpg', '961.jpg', '849.jpg', '559.jpg', '730.jpg', '917.jpg', '900.jpg', '73.jpg', '926.jpg', '416.jpg', '492.jpg', '284.jpg', '510.jpg', '146.jpg', '659.jpg', '473.jpg', '56.jpg', '152.jpg', '953.jpg', '147.jpg', '909.jpg', '338.jpg', '121.jpg', '636.jpg', '429.jpg', '605.jpg', '141.jpg', '54.jpg', '560.jpg', '74.jpg', '392.jpg', '654.jpg', '379.jpg', '912.jpg', '398.jpg', '950.jpg', '599.jpg', '417.jpg', '419.jpg', '999.jpg', '628.jpg', '680.jpg', '793.jpg', '632.jpg', '644.jpg', '693.jpg', '905.jpg', '281.jpg', '120.jpg', '679.jpg', '896.jpg', '137.jpg', '37.jpg', '528.jpg', '743.jpg', '573.jpg', '340.jpg', '681.jpg', '332.jpg', '951.jpg', '383.jpg', '773.jpg', '815.jpg', '482.jpg', '442.jpg', '86.jpg', '844.jpg', '29.jpg', '673.jpg', '661.jpg', '973.jpg', '434.jpg', '748.jpg', '581.jpg', '93.jpg', '745.jpg', '8.jpg', '151.jpg', '986.jpg', '639.jpg', '238.jpg', '952.jpg', '726.jpg', '784.jpg', '608.jpg', '850.jpg', '2.jpg', '875.jpg', '96.jpg', '872.jpg', '228.jpg', '554.jpg', '555.jpg', '719.jpg', '438.jpg', '403.jpg', '165.jpg', '571.jpg', '574.jpg', '292.jpg', '735.jpg', '551.jpg', '172.jpg', '358.jpg', '144.jpg', '802.jpg', '84.jpg', '451.jpg', '201.jpg', '25.jpg', '758.jpg', '556.jpg', '972.jpg', '82.jpg', '933.jpg', '161.jpg', '791.jpg', '421.jpg', '105.jpg', '514.jpg', '242.jpg', '28.jpg', '461.jpg', '94.jpg', '630.jpg', '364.jpg', '647.jpg', '439.jpg', '353.jpg', '739.jpg', '7.jpg', '650.jpg', '750.jpg', '825.jpg', '470.jpg', '304.jpg', '504.jpg', '691.jpg', '879.jpg', '717.jpg', '323.jpg', '307.jpg', '567.jpg', '889.jpg', '311.jpg', '211.jpg', '536.jpg', '775.jpg', '562.jpg', '247.jpg', '499.jpg', '590.jpg', '362.jpg', '475.jpg', '592.jpg', '675.jpg', '619.jpg', '215.jpg', '280.jpg', '718.jpg', '744.jpg', '617.jpg', '690.jpg', '445.jpg', '13.jpg', '78.jpg', '746.jpg', '832.jpg', '111.jpg', '820.jpg', '622.jpg', '217.jpg', '637.jpg', '343.jpg', '862.jpg', '848.jpg', '209.jpg', '45.jpg', '138.jpg', '155.jpg', '365.jpg', '566.jpg', '836.jpg', '366.jpg', '831.jpg', '544.jpg', '946.jpg', '139.jpg', '233.jpg', '256.jpg', '76.jpg', '806.jpg', '192.jpg', '648.jpg', '187.jpg', '397.jpg', '512.jpg', '386.jpg', '246.jpg', '786.jpg', '436.jpg', '149.jpg', '219.jpg', '274.jpg', '817.jpg', '320.jpg', '652.jpg', '494.jpg', '655.jpg', '593.jpg', '553.jpg', '772.jpg', '104.jpg', '695.jpg', '308.jpg', '568.jpg', '943.jpg', '548.jpg', '481.jpg', '374.jpg', '704.jpg', '910.jpg', '580.jpg', '771.jpg', '125.jpg', '928.jpg', '275.jpg', '497.jpg', '660.jpg', '549.jpg', '200.jpg', '38.jpg', '118.jpg', '20.jpg', '181.jpg', '824.jpg', '935.jpg', '993.jpg', '927.jpg', '537.jpg', '968.jpg', '47.jpg', '741.jpg', '270.jpg', '919.jpg', '272.jpg', '513.jpg', '477.jpg', '131.jpg', '779.jpg', '77.jpg', '479.jpg', '357.jpg', '252.jpg', '895.jpg', '495.jpg', '496.jpg', '240.jpg', '774.jpg', '624.jpg', '306.jpg', '689.jpg', '887.jpg', '732.jpg', '964.jpg', '231.jpg', '724.jpg', '485.jpg', '276.jpg', '764.jpg', '920.jpg', '916.jpg', '184.jpg', '884.jpg', '412.jpg', '216.jpg', '738.jpg', '929.jpg', '310.jpg', '868.jpg', '124.jpg', '612.jpg', '823.jpg', '949.jpg', '877.jpg', '91.jpg', '196.jpg', '698.jpg', '569.jpg', '208.jpg', '790.jpg', '818.jpg', '128.jpg', '865.jpg', '502.jpg', '220.jpg', '150.jpg', '236.jpg', '907.jpg', '263.jpg', '17.jpg', '404.jpg', '696.jpg', '688.jpg', '469.jpg', '437.jpg', '57.jpg', '100.jpg', '662.jpg', '932.jpg', '880.jpg', '902.jpg', '899.jpg', '585.jpg', '847.jpg', '752.jpg', '335.jpg', '563.jpg', '526.jpg', '9.jpg', '615.jpg', '640.jpg', '255.jpg', '941.jpg', '206.jpg', '409.jpg', '976.jpg', '789.jpg', '48.jpg', '108.jpg', '171.jpg', '807.jpg', '692.jpg', '545.jpg', '614.jpg', '394.jpg', '740.jpg', '98.jpg', '258.jpg', '69.jpg', '288.jpg', '177.jpg', '687.jpg', '232.jpg', '635.jpg', '101.jpg', '214.jpg', '345.jpg', '901.jpg', '709.jpg', '535.jpg', '95.jpg', '435.jpg', '936.jpg', '148.jpg', '657.jpg', '965.jpg', '377.jpg', '814.jpg', '455.jpg', '842.jpg', '625.jpg', '873.jpg', '333.jpg', '550.jpg', '794.jpg', '26.jpg', '959.jpg', '859.jpg', '808.jpg', '85.jpg', '16.jpg', '716.jpg', '351.jpg', '711.jpg', '21.jpg', '396.jpg', '369.jpg', '309.jpg', '6.jpg', '956.jpg', '588.jpg', '870.jpg', '450.jpg', '587.jpg', '646.jpg', '168.jpg', '401.jpg', '974.jpg', '97.jpg', '543.jpg', '81.jpg', '623.jpg', '103.jpg', '858.jpg', '466.jpg', '135.jpg', '835.jpg', '325.jpg', '424.jpg', '316.jpg', '418.jpg', '87.jpg', '783.jpg', '981.jpg', '31.jpg', '61.jpg', '991.jpg', '533.jpg', '613.jpg', '851.jpg', '116.jpg', '575.jpg', '452.jpg', '347.jpg', '67.jpg', '874.jpg', '294.jpg', '52.jpg', '674.jpg', '293.jpg', '780.jpg', '344.jpg', '59.jpg', '376.jpg', '297.jpg', '596.jpg', '194.jpg', '869.jpg', '963.jpg', '355.jpg', '253.jpg', '518.jpg', '449.jpg', '330.jpg', '997.jpg', '349.jpg', '609.jpg', '197.jpg', '714.jpg', '368.jpg', '122.jpg', '393.jpg', '982.jpg', '712.jpg', '410.jpg', '751.jpg', '668.jpg', '940.jpg', '160.jpg', '678.jpg', '947.jpg', '992.jpg', '792.jpg', '852.jpg', '157.jpg', '408.jpg', '143.jpg', '221.jpg', '480.jpg', '531.jpg', '426.jpg', '979.jpg', '799.jpg', '210.jpg', '507.jpg', '523.jpg', '620.jpg', '329.jpg', '839.jpg', '500.jpg', '185.jpg', '565.jpg', '753.jpg', '360.jpg', '169.jpg', '391.jpg', '129.jpg', '166.jpg', '586.jpg', '819.jpg', '539.jpg', '79.jpg', '757.jpg', '63.jpg', '203.jpg', '694.jpg', '207.jpg', '498.jpg', '649.jpg', '816.jpg', '611.jpg', '337.jpg', '778.jpg', '855.jpg', '484.jpg', '341.jpg', '826.jpg', '721.jpg', '922.jpg', '328.jpg', '728.jpg', '158.jpg', '671.jpg', '261.jpg', '491.jpg', '32.jpg', '467.jpg', '248.jpg', '300.jpg', '339.jpg', '411.jpg', '472.jpg', '140.jpg', '701.jpg', '109.jpg', '830.jpg', '727.jpg', '278.jpg', '530.jpg']
# 
#
# %%
import os
import shutil
import torch
from torchvision import datasets, transforms
import splitfolders
import cv2
from torch.utils.data import Dataset
from PIL import Image  # PIL ?쇱씠釉뚮윭由?異붽?

# train ?곗씠?곕? 8:2濡?遺꾪븷
train_dir = '/content/train'
split2_dir = '/content/split2_train_val'
splitfolders.ratio(train_dir, output=split2_dir, ratio=(.8, .2), group_prefix=None)

# train, val, test ?곗씠???붾젆?좊━ ?ㅼ젙
train_data_dir = '/content/split2_train_val/train'
val_data_dir = '/content/split2_train_val/val'
test_data_dir = '/content/test'

# train ?대뜑 ?댁쓽 遺덊븘?뷀븳 'train' ?대뜑 ?쒓굅
for class_name in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_name)
    if os.path.isdir(class_path) and class_name == 'train':
        shutil.rmtree(class_path)  # 'train' ?대뜑 ??젣

for class_name in os.listdir(val_data_dir):
    class_path = os.path.join(val_data_dir, class_name)
    if os.path.isdir(class_path) and class_name == 'train':
        shutil.rmtree(class_path)  # 'train' ?대뜑 ??젣

# OpenCV濡??대?吏瑜?濡쒕뱶?섍퀬 蹂?섑븯??Dataset ?대옒???뺤쓽
class OpenCVDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []

        # ?대?吏 寃쎈줈 由ъ뒪??留뚮뱾湲?        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append((os.path.join(class_dir, filename), class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
        img = cv2.imread(img_path)  # OpenCV濡??대?吏 ?쎄린
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR??RGB濡?蹂??        img = Image.fromarray(img)  # NumPy 諛곗뿴??PIL ?대?吏濡?蹂??
        if self.transform:
            img = self.transform(img)  # ?꾩슂??蹂???곸슜

        # ?대옒???대쫫???뺤닔濡?蹂??        label = self.classes.index(class_name)

        return img, label

# ?곗씠??蹂???뺤쓽
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

# ?곗씠?곗뀑 濡쒕뵫
train_dataset = OpenCVDataset(train_data_dir, transform=data_transforms['train'])
val_dataset = OpenCVDataset(val_data_dir, transform=data_transforms['val'])
test_dataset = OpenCVDataset(test_data_dir, transform=data_transforms['test'])

# ?곗씠?곕줈???ㅼ젙
batch_size = 16
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ?곗씠?곗뀑 ?ш린
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

# ?대옒???대쫫
class_names = train_dataset.classes

# GPU ?ъ슜 ?щ????곕씪 device ?ㅼ젙
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 異쒕젰 ?뺤씤
print(f"Train size: {dataset_sizes['train']}")
print(f"Validation size: {dataset_sizes['val']}")
print(f"Test size: {dataset_sizes['test']}")
print(f"Class names: {class_names}")
print(f"Device: {device}")
# %% [markdown]
# Saved output
# Copying files: 10569 files [00:02, 4589.97 files/s]
# 
# Train size: 8452
# Validation size: 2117
# Test size: 1000
# Class names: ['kangdaniel', 'skz_hunjin', 'bts_v', 'Kimyuna', 'ive_jangwonyoung', 'IU', 'bts_jimin', 'astro_chaeunwoo', 'aespa_karina', 'rv_irene']
# Device: cuda:0
# 
#
# %%
import os
import random
import cv2
import matplotlib.pyplot as plt


# ?쒕툕?뚮’ ?ㅼ젙 (2x5 ?뺥깭)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()  # 2D 諛곗뿴??1D濡?蹂??
# 媛??대옒?ㅻ쭏???섎굹???대?吏 ?좏깮?섏뿬 異쒕젰
for idx, class_name in enumerate(class_names[:10]):  # 泥?10媛??대옒?ㅻ쭔 泥섎━
    # ?대옒???붾젆?좊━ ?댁쓽 ?대?吏 ?뚯씪??    class_dir = os.path.join(train_data_dir, class_name)
    image_files = os.listdir(class_dir)

    # ?쒕뜡?쇰줈 ?섎굹???대?吏 ?좏깮
    image_path = os.path.join(class_dir, random.choice(image_files))

    # OpenCV瑜??ъ슜?섏뿬 ?대?吏 ?닿린
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR?먯꽌 RGB濡?蹂??
    # ?쒕툕?뚮’???대?吏 異쒕젰
    axes[idx].imshow(img)
    axes[idx].axis('off')  # 異뺤쓣 ?쒖떆?섏? ?딆쓬
    axes[idx].set_title(class_name)  # ?대옒???대쫫 ?쒖떆

# ?쒕툕?뚮’ 媛꾧꺽 議곗젙
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1500x600 with 10 Axes>
#
# %%
# ?덈젴, 寃利? ?뚯뒪???곗씠?곗뀑 ?ш린 ?뺤씤
print(f"Train dataset size: {dataset_sizes['train']}")
print(f"Validation dataset size: {dataset_sizes['val']}")
print(f"Test dataset size: {dataset_sizes['test']}")
# %% [markdown]
# Saved output
# Train dataset size: 8452
# Validation dataset size: 2117
# Test dataset size: 1000
# 
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
# ### **1. ?쇨뎬?몄떇 ?뚭퀬由ъ쬁 鍮꾧탳**
#
# %% [markdown]
# 
# - ResNet50: ?쇰컲?곸씤 ?대?吏 遺꾨쪟 紐⑤뜽濡??쇨뎬 ?몄떇?먮룄 ?ъ슜 媛?ν빀?덈떎.
# - VGGFace (ResNet50): VGGFace ?쇱씠釉뚮윭由ъ뿉???쒓났?섎뒗 ResNet50 湲곕컲 紐⑤뜽.
# - FaceNet: ?쇨뎬 ?몄떇???뱁솕??紐⑤뜽濡? DeepFace ?쇱씠釉뚮윭由ъ뿉???쒓났.
# - EfficientNet: ?⑥쑉?곸씤 ?깅뒫???쒓났?섎뒗 紐⑤뜽濡??쇨뎬 ?몄떇?먮룄 ?ъ슜 媛?ν빀?덈떎.
# - MobileNetV2: 寃쎈웾?붾맂 紐⑤뜽濡?鍮좊Ⅴ怨??⑥쑉?곸씤 ?쇨뎬 ?몄떇???곹빀?⑸땲??
# - ArcFace: ?쇨뎬 ?몄떇???뱁솕??紐⑤뜽濡? 留ㅼ슦 ?믪? ?뺥솗?꾨? ?먮옉?⑸땲??
#
# %%
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch

# Mixed precision training ?ъ슜
scaler = GradScaler()

def train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    file_name = f"{model_name}_model_params.pt"
    best_acc = 0.0
    loss_dict = {"train": [], "val": []}  # ?먯떎 ????뺤뀛?덈━
    acc_dict = {"train": [], "val": []}   # ?뺥솗??????뺤뀛?덈━

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # ?숈뒿 紐⑤뱶
            else:
                model.eval()    # ?됯? 紐⑤뱶

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()     # ?듯떚留덉씠? 湲곗슱湲?0?쇰줈 ?ㅼ젙

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():  # Mixed precision training ?쒖꽦??                        outputs = model(inputs)
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

            # ?먯떎 洹몃옒?????            plt.plot(range(len(loss_dict[phase])), loss_dict[phase])
            plt.title(f"{model_name} - {phase} Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f"{model_name}_{phase}_Loss.png")
            plt.close()

            # ?뺥솗??洹몃옒?????            plt.plot(range(len(acc_dict[phase])), acc_dict[phase])
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
                scheduler.step(epoch_loss)   # ?ㅼ?伊대윭 ?ъ슜

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    # 異뷀썑 洹몃옒?꾨? 異쒕젰?섍린 ?꾪빐 ?먯떎/?뺥솗??湲곕줉
    torch.save(loss_dict, f"{model_name}_loss_dict.pth")
    torch.save(acc_dict, f"{model_name}_acc_dict.pth")
# %% [markdown]
# Saved output
# <ipython-input-9-ed3bb1437506>:6: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
#   scaler = GradScaler()
# 
#
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
# %%
loss_function = torch.nn.CrossEntropyLoss() # loss function ?뺤쓽
max_epochs = 10 # epoch ???뺤쓽
n_features = len(class_names)
# %% [markdown]
# 
# ####**1. resnet50**
#
# %%
model_1 = models.resnet50(pretrained=True)
model_1.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_1.fc.in_features, n_features)
)
model_1.to(device)

optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_1, 'resnet50', loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# Epoch 1/10
# ----------
# train Total Loss: 2.0324 Acc: 0.2550
# val Total Loss: 1.5761 Acc: 0.4747
# 
# Epoch 2/10
# ----------
# train Total Loss: 1.4623 Acc: 0.4947
# val Total Loss: 1.1373 Acc: 0.6160
# 
# Epoch 3/10
# ----------
# train Total Loss: 1.1572 Acc: 0.5996
# val Total Loss: 0.9682 Acc: 0.6594
# 
# Epoch 4/10
# ----------
# train Total Loss: 0.9770 Acc: 0.6721
# val Total Loss: 0.9096 Acc: 0.6793
# 
# Epoch 5/10
# ----------
# train Total Loss: 0.8637 Acc: 0.7042
# val Total Loss: 0.7936 Acc: 0.7204
# 
# Epoch 6/10
# ----------
# train Total Loss: 0.7717 Acc: 0.7384
# val Total Loss: 0.7264 Acc: 0.7444
# 
# Epoch 7/10
# ----------
# train Total Loss: 0.6882 Acc: 0.7677
# val Total Loss: 0.6888 Acc: 0.7681
# 
# Epoch 8/10
# ----------
# train Total Loss: 0.6301 Acc: 0.7841
# val Total Loss: 0.6659 Acc: 0.7770
# 
# Epoch 9/10
# ----------
# train Total Loss: 0.5695 Acc: 0.8058
# val Total Loss: 0.6317 Acc: 0.7893
# 
# Epoch 10/10
# ----------
# train Total Loss: 0.5279 Acc: 0.8238
# val Total Loss: 0.6270 Acc: 0.7884
# 
# Best val Acc: 0.7893
# 
#
# %% [markdown]
# 
# ####**2. EfficientNet**
#
# %%
model_2 = EfficientNet.from_pretrained('efficientnet-b0').to(device)
model_2._fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model_2._fc.in_features, n_features)
)
model_2.to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-5)
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_2, 'efficientnet', loss_function, optimizer_2, scheduler_2, max_epochs, dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# Loaded pretrained weights for efficientnet-b0
# Epoch 1/10
# ----------
# train Total Loss: 2.2956 Acc: 0.1227
# val Total Loss: 2.2440 Acc: 0.1904
# 
# Epoch 2/10
# ----------
# train Total Loss: 2.1913 Acc: 0.1973
# val Total Loss: 2.1135 Acc: 0.2985
# 
# Epoch 3/10
# ----------
# train Total Loss: 2.0546 Acc: 0.2764
# val Total Loss: 1.9366 Acc: 0.3769
# 
# Epoch 4/10
# ----------
# train Total Loss: 1.9115 Acc: 0.3261
# val Total Loss: 1.7823 Acc: 0.4218
# 
# Epoch 5/10
# ----------
# train Total Loss: 1.7910 Acc: 0.3720
# val Total Loss: 1.6530 Acc: 0.4676
# 
# Epoch 6/10
# ----------
# train Total Loss: 1.6720 Acc: 0.4164
# val Total Loss: 1.5373 Acc: 0.4955
# 
# Epoch 7/10
# ----------
# train Total Loss: 1.5687 Acc: 0.4503
# val Total Loss: 1.4358 Acc: 0.5172
# 
# Epoch 8/10
# ----------
# train Total Loss: 1.4821 Acc: 0.4852
# val Total Loss: 1.3427 Acc: 0.5413
# 
# Epoch 9/10
# ----------
# train Total Loss: 1.4085 Acc: 0.5022
# val Total Loss: 1.2666 Acc: 0.5664
# 
# Epoch 10/10
# ----------
# train Total Loss: 1.3414 Acc: 0.5288
# val Total Loss: 1.2083 Acc: 0.5820
# 
# Best val Acc: 0.5820
# 
#
# %% [markdown]
# 
# ####**3. MobileNetV2**
#
# %%
from torchvision import models

model_3 = models.mobilenet_v2(pretrained=True).to(device)
model_3.classifier[1] = nn.Linear(model_3.classifier[1].in_features, n_features)
model_3.to(device)

optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=1e-5)
scheduler_3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_3, mode='min', factor=0.1, patience=10, verbose=True)

train_model(model_3, 'mobilenetv2', loss_function, optimizer_3, scheduler_3, max_epochs, dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# Epoch 1/10
# ----------
# train Total Loss: 2.1169 Acc: 0.2320
# val Total Loss: 1.8209 Acc: 0.3618
# 
# Epoch 2/10
# ----------
# train Total Loss: 1.7182 Acc: 0.3978
# val Total Loss: 1.5433 Acc: 0.4521
# 
# Epoch 3/10
# ----------
# train Total Loss: 1.4963 Acc: 0.4831
# val Total Loss: 1.3602 Acc: 0.5078
# 
# Epoch 4/10
# ----------
# train Total Loss: 1.3484 Acc: 0.5316
# val Total Loss: 1.2468 Acc: 0.5621
# 
# Epoch 5/10
# ----------
# train Total Loss: 1.2433 Acc: 0.5709
# val Total Loss: 1.1518 Acc: 0.5919
# 
# Epoch 6/10
# ----------
# train Total Loss: 1.1665 Acc: 0.5939
# val Total Loss: 1.1014 Acc: 0.6042
# 
# Epoch 7/10
# ----------
# train Total Loss: 1.1075 Acc: 0.6125
# val Total Loss: 1.0251 Acc: 0.6372
# 
# Epoch 8/10
# ----------
# train Total Loss: 1.0590 Acc: 0.6275
# val Total Loss: 0.9992 Acc: 0.6419
# 
# Epoch 9/10
# ----------
# train Total Loss: 0.9971 Acc: 0.6505
# val Total Loss: 0.9449 Acc: 0.6693
# 
# Epoch 10/10
# ----------
# train Total Loss: 0.9520 Acc: 0.6649
# val Total Loss: 0.9217 Acc: 0.6693
# 
# Best val Acc: 0.6693
# 
#
# %% [markdown]
# 
# ####**4. InceptionV3**
#
# %%
def inception_train_model(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')
        print('-' * 10)

        # 媛?epoch留덈떎 ?덈젴怨?寃利??④퀎 ?섑뻾
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # ?곗씠??濡쒕뵫
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # 寃쎌궗 珥덇린??                optimizer.zero_grad()

                with autocast():  # Mixed precision training ?쒖꽦??                    outputs = model(inputs)
                    # outputs[0]? logits?낅땲??
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs  # outputs媛 ?쒗뵆???뚮쭔 [0]???ъ슜
                    _, preds = torch.max(logits, 1)  # logits???ъ슜?섏뿬 ?덉륫媛?異붿텧
                    loss = loss_function(logits, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 寃利??깅뒫??媛쒖꽑?섎㈃ 紐⑤뜽 ???            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # ?숈뒿瑜??ㅼ?以꾨윭 ?낅뜲?댄듃
        scheduler.step(epoch_loss)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model
# %%
torch.cuda.empty_cache()
# %%
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
    transforms.Resize((299, 299)),  # InceptionV3???낅젰 ?ш린
    transforms.ToTensor(),          # ?먯꽌濡?蹂??    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ?뺢퇋??])

inception_train_dataset = OpenCVDataset(train_data_dir, transform=inception_transform)  # InceptionV3??留욌뒗 transform ?ъ슜
inception_val_dataset = OpenCVDataset(val_data_dir, transform=inception_transform)      # InceptionV3??留욌뒗 transform ?ъ슜
inception_test_dataset = OpenCVDataset(test_data_dir, transform=inception_transform)    # InceptionV3??留욌뒗 transform ?ъ슜

# ?곗씠?곕줈???ㅼ젙
batch_size = 16
inception_dataloaders = {
    'train': torch.utils.data.DataLoader(inception_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(inception_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(inception_test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ?듯떚留덉씠?? ?ㅼ?以꾨윭 ?ㅼ젙
optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=1e-5)
scheduler_4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_4, mode='min', factor=0.1, patience=10, verbose=True)

# 紐⑤뜽 ?덈젴
inception_train_model(model_4, 'inception_v3', loss_function, optimizer_4, scheduler_4, max_epochs, inception_dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# Epoch 1/10
# ----------
# 
# <ipython-input-14-0a8541c1a20c>:21: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with autocast():  # Mixed precision training ?쒖꽦??# 
# train Loss: 2.2080 Acc: 0.1850
# val Loss: 1.8948 Acc: 0.4133
# Epoch 2/10
# ----------
# train Loss: 1.7609 Acc: 0.3891
# val Loss: 1.4819 Acc: 0.5352
# Epoch 3/10
# ----------
# train Loss: 1.3562 Acc: 0.5600
# val Loss: 1.1514 Acc: 0.6311
# Epoch 4/10
# ----------
# train Loss: 1.0122 Acc: 0.6882
# val Loss: 0.9339 Acc: 0.6963
# Epoch 5/10
# ----------
# train Loss: 0.7353 Acc: 0.7836
# val Loss: 0.8030 Acc: 0.7308
# Epoch 6/10
# ----------
# train Loss: 0.5420 Acc: 0.8444
# val Loss: 0.7297 Acc: 0.7572
# Epoch 7/10
# ----------
# train Loss: 0.3777 Acc: 0.9003
# val Loss: 0.6982 Acc: 0.7615
# Epoch 8/10
# ----------
# train Loss: 0.2744 Acc: 0.9327
# val Loss: 0.6958 Acc: 0.7780
# Epoch 9/10
# ----------
# train Loss: 0.1964 Acc: 0.9548
# val Loss: 0.7033 Acc: 0.7770
# Epoch 10/10
# ----------
# train Loss: 0.1378 Acc: 0.9741
# val Loss: 0.7291 Acc: 0.7690
# Best val Acc: 0.7780
# 
# Inception3(
#   (Conv2d_1a_3x3): BasicConv2d(
#     (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#     (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (Conv2d_2a_3x3): BasicConv2d(
#     (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
#     (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (Conv2d_2b_3x3): BasicConv2d(
#     (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (Conv2d_3b_1x1): BasicConv2d(
#     (conv): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     (bn): BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (Conv2d_4a_3x3): BasicConv2d(
#     (conv): Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)
#     (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#   )
#   (maxpool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (Mixed_5b): InceptionA(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_1): BasicConv2d(
#       (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_2): BasicConv2d(
#       (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3): BasicConv2d(
#       (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_5c): InceptionA(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_1): BasicConv2d(
#       (conv): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_2): BasicConv2d(
#       (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3): BasicConv2d(
#       (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_5d): InceptionA(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_1): BasicConv2d(
#       (conv): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch5x5_2): BasicConv2d(
#       (conv): Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3): BasicConv2d(
#       (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_6a): InceptionB(
#     (branch3x3): BasicConv2d(
#       (conv): Conv2d(288, 384, kernel_size=(3, 3), stride=(2, 2), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3): BasicConv2d(
#       (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
#       (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_6b): InceptionC(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_1): BasicConv2d(
#       (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_2): BasicConv2d(
#       (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_3): BasicConv2d(
#       (conv): Conv2d(128, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_1): BasicConv2d(
#       (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_2): BasicConv2d(
#       (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_3): BasicConv2d(
#       (conv): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_4): BasicConv2d(
#       (conv): Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_5): BasicConv2d(
#       (conv): Conv2d(128, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_6c): InceptionC(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_1): BasicConv2d(
#       (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_2): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_3): BasicConv2d(
#       (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_1): BasicConv2d(
#       (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_2): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_3): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_4): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_5): BasicConv2d(
#       (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_6d): InceptionC(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_1): BasicConv2d(
#       (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_2): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_3): BasicConv2d(
#       (conv): Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_1): BasicConv2d(
#       (conv): Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_2): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_3): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_4): BasicConv2d(
#       (conv): Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_5): BasicConv2d(
#       (conv): Conv2d(160, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_6e): InceptionC(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_2): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7_3): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_2): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_3): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_4): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7dbl_5): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (AuxLogits): InceptionAux(
#     (conv0): BasicConv2d(
#       (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (conv1): BasicConv2d(
#       (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (fc): Sequential(
#       (0): Dropout(p=0.5, inplace=False)
#       (1): Linear(in_features=768, out_features=10, bias=True)
#     )
#   )
#   (Mixed_7a): InceptionD(
#     (branch3x3_1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_2): BasicConv2d(
#       (conv): Conv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)
#       (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7x3_1): BasicConv2d(
#       (conv): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7x3_2): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7x3_3): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch7x7x3_4): BasicConv2d(
#       (conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_7b): InceptionE(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_1): BasicConv2d(
#       (conv): Conv2d(1280, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_2a): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_2b): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(1280, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3a): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3b): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(1280, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (Mixed_7c): InceptionE(
#     (branch1x1): BasicConv2d(
#       (conv): Conv2d(2048, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_1): BasicConv2d(
#       (conv): Conv2d(2048, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_2a): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3_2b): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_1): BasicConv2d(
#       (conv): Conv2d(2048, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(448, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_2): BasicConv2d(
#       (conv): Conv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3a): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch3x3dbl_3b): BasicConv2d(
#       (conv): Conv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False)
#       (bn): BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (branch_pool): BasicConv2d(
#       (conv): Conv2d(2048, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (dropout): Dropout(p=0.5, inplace=False)
#   (fc): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=2048, out_features=10, bias=True)
#   )
# )
#
# %%
import cv2
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# ?쇨뎬 寃異??⑥닔 (Haar Cascade Classifier ?ъ슜)
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# ?쇨뎬 寃異????숈뒿 ?⑥닔
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

# ResNet50 紐⑤뜽 ?ㅼ젙
model_5 = models.resnet50(pretrained=True)
model_5.fc = nn.Linear(model_5.fc.in_features, n_features)  # ?대옒???섏뿉 留욊쾶 異쒕젰痢??섏젙
model_5.to(device)

optimizer_5 = torch.optim.Adam(model_5.parameters(), lr=1e-5)
scheduler_5 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_5, mode='min', factor=0.1, patience=10, verbose=True)

train_with_face_detection_haar(model_5, 'resnet50_haar', loss_function, optimizer_5, scheduler_5, max_epochs, dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# /usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# 
# Epoch 1/10
# ----------
# 
# <ipython-input-25-0a12050fe90f>:41: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with autocast():
# 
# train Loss: 1.8841 Acc: 0.3499
# val Loss: 1.4150 Acc: 0.5177
# Epoch 2/10
# ----------
# 
# KeyboardInterrupt: 
#
# %%
import cv2
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# ?쇨뎬 寃異??⑥닔 (Haar Cascade Classifier ?ъ슜)
def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGB -> Grayscale濡?蹂??    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# InceptionV3 ?꾩슜 ?쇨뎬 寃異??덈젴 ?⑥닔
def inception_train_with_face_detection_haar(model, model_name, loss_function, optimizer, scheduler, max_epochs, dataloaders, dataset_sizes, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')
        print('-' * 10)

        # 媛?epoch留덈떎 ?덈젴怨?寃利??④퀎 ?섑뻾
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # ?곗씠??濡쒕뵫
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # 寃쎌궗 珥덇린??                optimizer.zero_grad()

                with autocast():  # Mixed precision training ?쒖꽦??                    # ?쇨뎬 寃異??⑥닔 ?곸슜
                    for i in range(inputs.size(0)):  # 諛곗튂留덈떎 ?쇨뎬 寃異?                        img = inputs[i].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                        img = (img * 255).astype('uint8')  # ?먯꽌瑜?0-255 踰붿쐞濡?蹂?섑븯??OpenCV?먯꽌 泥섎━ 媛?ν븯寃???                        faces = detect_faces_haar(img)
                        # ?쇨뎬 寃異???異붽??곸씤 泥섎━瑜??????덉뒿?덈떎 (?? ?쇨뎬 ?곸뿭留??섎씪???숈뒿)
                        # ??遺遺꾩? ?꾩슂???곕씪 ?섏젙 媛?ν빀?덈떎.

                    # 紐⑤뜽???덉륫 ?섑뻾
                    outputs = model(inputs)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs  # outputs媛 ?쒗뵆???뚮쭔 [0]???ъ슜
                    _, preds = torch.max(logits, 1)  # logits???ъ슜?섏뿬 ?덉륫媛?異붿텧
                    loss = loss_function(logits, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 寃利??깅뒫??媛쒖꽑?섎㈃ 紐⑤뜽 ???            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # ?숈뒿瑜??ㅼ?以꾨윭 ?낅뜲?댄듃
        scheduler.step(epoch_loss)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# model_7 ?ㅼ젙 (InceptionV3 紐⑤뜽)
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

# ?곗씠?곗뀑??留욌뒗 transform ?ㅼ젙
inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3???낅젰 ?ш린
    transforms.ToTensor(),          # ?먯꽌濡?蹂??    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ?뺢퇋??])

# ?곗씠?곗뀑 ?ㅼ젙
inception_train_dataset = OpenCVDataset(train_data_dir, transform=inception_transform)
inception_val_dataset = OpenCVDataset(val_data_dir, transform=inception_transform)
inception_test_dataset = OpenCVDataset(test_data_dir, transform=inception_transform)

# ?곗씠?곕줈???ㅼ젙
batch_size = 16
inception_dataloaders = {
    'train': torch.utils.data.DataLoader(inception_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
    'val': torch.utils.data.DataLoader(inception_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    'test': torch.utils.data.DataLoader(inception_test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
}

# ?듯떚留덉씠?? ?ㅼ?以꾨윭 ?ㅼ젙
optimizer_7 = torch.optim.Adam(model_7.parameters(), lr=1e-5)
scheduler_7 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_7, mode='min', factor=0.1, patience=10, verbose=True)

# 紐⑤뜽 ?덈젴
inception_train_with_face_detection_haar(model_7, 'inception_v3_haar', loss_function, optimizer_7, scheduler_7, max_epochs, inception_dataloaders, dataset_sizes, device)
# %% [markdown]
# Saved output
# Epoch 1/10
# ----------
# 
# <ipython-input-23-49df428d45ac>:37: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with autocast():  # Mixed precision training ?쒖꽦??# 
# train Loss: 2.1958 Acc: 0.1854
# val Loss: 1.8879 Acc: 0.3855
# Epoch 2/10
# ----------
# train Loss: 1.7627 Acc: 0.3797
# val Loss: 1.5125 Acc: 0.5083
# Epoch 3/10
# ----------
# train Loss: 1.3954 Acc: 0.5351
# val Loss: 1.1858 Acc: 0.6094
# Epoch 4/10
# ----------
# train Loss: 1.0557 Acc: 0.6668
# val Loss: 0.9570 Acc: 0.6863
# Epoch 5/10
# ----------
# train Loss: 0.7818 Acc: 0.7641
# val Loss: 0.8182 Acc: 0.7251
# Epoch 6/10
# ----------
# train Loss: 0.5511 Acc: 0.8415
# val Loss: 0.7564 Acc: 0.7359
# Epoch 7/10
# ----------
# train Loss: 0.4037 Acc: 0.8930
# val Loss: 0.7205 Acc: 0.7624
# Epoch 8/10
# ----------
# train Loss: 0.2774 Acc: 0.9309
# val Loss: 0.7222 Acc: 0.7638
# Epoch 9/10
# ----------
# 
# KeyboardInterrupt: 
#
# %% [markdown]
# 
# 
#
# %%
import matplotlib.pyplot as plt
import torch

plt.figure(figsize=(12, 8))

# ResNet50
resnet50_acc_dict = torch.load("/content/resnet50_acc_dict.pth")
plt.plot(range(1, len(resnet50_acc_dict['val']) + 1), resnet50_acc_dict['val'], label='ResNet50')

# ResNet50 with Haar
resnet50_haar_acc_dict = torch.load("/content/resnet50_haar_acc_dict.pth")
plt.plot(range(1, len(resnet50_haar_acc_dict['val']) + 1), resnet50_haar_acc_dict['val'], label='ResNet50 (Haar)')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy of Models')
plt.savefig('validation_accuracy.png')
plt.show()
# %% [markdown]
# Saved output
# <ipython-input-24-40491e6a1cd6>:7: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   resnet50_acc_dict = torch.load("/content/resnet50_acc_dict.pth")
# <ipython-input-24-40491e6a1cd6>:11: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   resnet50_haar_acc_dict = torch.load("/content/resnet50_haar_acc_dict.pth")
# 
# FileNotFoundError: [Errno 2] No such file or directory: '/content/resnet50_haar_acc_dict.pth'
# <Figure size 1200x800 with 1 Axes>
#
# %%
import random
import matplotlib.pyplot as plt
import numpy as np

# 紐⑤뜽???됯? 紐⑤뱶濡??ㅼ젙
model_7.eval()

# 媛??대옒?ㅼ뿉???쒕뜡?쇰줈 ?대?吏瑜??좏깮
selected_images = []
for class_name in class_names:
    class_dir = os.path.join(test_data_dir, class_name)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    selected_images.append(random.choice(images))

# 2x5 ?쒕툕?뚮’ ?앹꽦
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for idx, img_path in enumerate(selected_images):
    # ?대?吏 濡쒕뱶 諛?蹂??    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transforms['test'](img).unsqueeze(0).to(device)

    # 紐⑤뜽 ?덉륫
    with torch.no_grad():
        outputs = model_7(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # ?쒓컖??    img_np = np.array(img)
    axes[idx].imshow(img_np)
    axes[idx].axis('off')
    axes[idx].set_title(f"{class_names[predicted_class]}\n{confidence:.2f}")

# ?덉씠?꾩썐 議곗젙
plt.tight_layout()
plt.show()
# %% [markdown]
# Saved output
# <Figure size 1500x600 with 10 Axes>
#
# %% [markdown]
# 
# 
#
# %% [markdown]
# 
# 
#
# %% [markdown]
# 
# 
#
# %% [markdown]
# 
# 
#
# %% [markdown]
# 
# ####**model_1~5 Evaluation**
#
# %%
model_best=model_8
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

