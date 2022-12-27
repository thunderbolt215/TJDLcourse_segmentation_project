import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CityscapeDataset
from network import UNet


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

data_dir = os.path.join("cityscapes-image-pairs", "cityscapes_data")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)


# analyze data
sample_image_fp = os.path.join(train_dir, train_fns[0])
sample_image = Image.open(sample_image_fp).convert("RGB")


def split_image(image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label


sample_image = np.array(sample_image)
cityscape, label = split_image(sample_image)
cityscape, label = Image.fromarray(cityscape), Image.fromarray(label)


# Define labels
num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

num_classes = 10
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)

label_model.predict(color_array[:5, :])

cityscape, label = split_image(sample_image)
label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)

# hyperparameters
batch_size = 8
epochs = 10
lr = 0.01
dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size=batch_size)
model = UNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# step_losses = []
# epoch_losses = []
# for epoch in tqdm(range(epochs)):
#     epoch_loss = 0
#     for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
#         X, Y = X.to(device), Y.to(device)
#         optimizer.zero_grad()
#         Y_pred = model(X)
#         loss = criterion(Y_pred, Y)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         step_losses.append(loss.item())
#     epoch_losses.append(epoch_loss/len(data_loader))

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].plot(step_losses)
# axes[0].title('step_losses')
# axes[1].plot(epoch_losses)
# axes[1].title('epoch_losses')
# plt.figure()
# plt.title('step_losses')
# plt.plot(step_losses)
# plt.xlabel('step')
# plt.ylabel('loss')
# plt.show()

# plt.figure()
# plt.title('epoch_losses')
# plt.plot(epoch_losses)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()



# test
model_path = "MyUNet.pth"
model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

test_batch_size = 8
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
# print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
# print(Y_pred.shape)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

# fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))
for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(landscape)
    plt.title("Landscape")
    plt.subplot(132)
    plt.imshow(label_class)
    plt.title("Label Class")
    plt.subplot(133)
    plt.imshow(label_class_predicted)
    plt.title("Label Class - Predicted")
    plt.tight_layout()
    plt.show()

