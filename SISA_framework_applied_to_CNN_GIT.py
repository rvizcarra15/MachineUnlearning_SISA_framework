# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:07:02 2024

@author: https://www.linkedin.com/in/raulvizcarrach
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#******************************Data transformation********************************************
# Training and Validation Datasets
data_dir = 'D:/PYTHON/teams_sample_dataset'

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load data
full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

#******************************Sharding the dataset**************************

def shard_dataset(dataset, num_shards):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    shards = []
    shard_size = len(dataset) // num_shards
    for i in range(num_shards):
        shard_indices = indices[i * shard_size : (i + 1) * shard_size]
        shards.append(Subset(dataset, shard_indices))
    return shards

#******************************Overlapping Slices***************************
def create_overlapping_slices(shard, slice_size, overlap):
    indices = list(shard.indices)
    slices = []
    step = slice_size - overlap
    for start in range(0, len(indices) - slice_size + 1, step):
        slice_indices = indices[start:start + slice_size]
        slices.append(Subset(shard.dataset, slice_indices))
    return slices

#**************************Applying Sharding and Slicing*******************

num_shards = 4  
slice_size = len(full_train_dataset) // num_shards // 2
overlap = slice_size // 2
shards = shard_dataset(full_train_dataset, num_shards)

#************************Overlapping slices for each shard*****************
all_slices = []
for shard in shards:
    slices = create_overlapping_slices(shard, slice_size, overlap)
    all_slices.extend(slices)

#**************************+*Isolate datapoints******************************
def isolate_data_for_unlearning(slice, data_points_to_remove):
    new_indices = [i for i in slice.indices if i not in data_points_to_remove]
    return Subset(slice.dataset, new_indices)

#*****Identify the indices of the images we want to rectify/erasure**********
def get_indices_to_remove(dataset, image_names_to_remove):
    indices_to_remove = []
    image_to_index = {img_path: idx for idx, (img_path, _) in enumerate(dataset.imgs)}
    for image_name in image_names_to_remove:
        if image_name in image_to_index:
            indices_to_remove.append(image_to_index[image_name])
    return indices_to_remove

#*************************Specify and remove images***************************
images_to_remove = ["Away_image03.JPG", "Away_image04.JPG", "Away_image05.JPG"]
indices_to_remove = get_indices_to_remove(full_train_dataset, images_to_remove)
updated_slices = [isolate_data_for_unlearning(slice, indices_to_remove) for slice in all_slices]

#********************************CNN Model Architecture**************************************

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  # Output three classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#********************************CNN TRAINING**********************************************

# Model-loss function-optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#*********************************Training*************************************************
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for slice in updated_slices:
        train_loader = DataLoader(slice, batch_size=32, shuffle=True)
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    train_losses.append(running_loss / (len(updated_slices)))

    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        for inputs, labels in val_loader:
            outputs = model(inputs)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

#********************************METRICS & PERFORMANCE************************************
    
    val_losses.append(val_loss / len(val_loader))
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, "
          f"Val Acc: {val_accuracy:.2%}, "
          f"Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, "
          f"Val F1 Score: {val_f1:.4f}")

#*******************************SHOW METRICS & PERFORMANCE**********************************
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

# SAVE THE MODEL FOR THE GH_CV_track_teams CODE
torch.save(model.state_dict(), 'hockey_team_classifier_SISA.pth')