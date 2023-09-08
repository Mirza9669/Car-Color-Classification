import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import random
import tensorflow as tf
from torchvision import transforms, models
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

csv_folder = r'H:\Internship\Car Classification\CSV'

train_df = pd.read_csv(r'H:\Internship\Car Classification\CSV\train.csv')
test_df = pd.read_csv(r'H:\Internship\Car Classification\CSV\test.csv')
val_df = pd.read_csv(r'H:\Internship\Car Classification\CSV\val.csv')

train_img_folder = r'H:\Internship\Car Classification\Dataset\train'
test_img_folder = r'H:\Internship\Car Classification\Dataset\test'
val_img_folder = r'H:\Internship\Car Classification\Dataset\val'

print(train_df)
print(test_df)
print(val_df)

data_transforms = {
    'train' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'test' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'val' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])
}

def load_image(img_path, transform=data_transforms):
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

train_images = []
previous_label = None
break_index = 0

for file in os.listdir(train_img_folder):
    print(file)    
    for index in range(break_index, len(train_df)):
        img = train_df.iloc[index, 0]
        label = train_df.iloc[index, 1]
        
        if label != previous_label and previous_label is not None:
            break_index = index
            previous_label = label
            break  
            
        image_path = os.path.join(os.path.join(train_img_folder, file), img + '.jpg')
        train_image = load_image(image_path, transform=data_transforms['train'])
        train_images.append(train_image)
        
        previous_label = label

print(f"Train Images Count: {len(train_images)}")

test_images = []
previous_label = None
break_index = 0

for file in os.listdir(test_img_folder):
    print(file)    
    for index in range(break_index, len(test_df)):
        img = test_df.iloc[index, 0]
        label = test_df.iloc[index, 1]
        
        if label != previous_label and previous_label is not None:
            break_index = index
            previous_label = label
            break  
            
        image_path = os.path.join(os.path.join(test_img_folder, file), img + '.jpg')
        test_image = load_image(image_path, transform=data_transforms['test'])
        test_images.append(test_image)
        
        previous_label = label

print(f"Test Images Count: {len(test_images)}")

val_images = []
previous_label = None
break_index = 0

for file in os.listdir(val_img_folder):
    print(file)    
    for index in range(break_index, len(val_df)):
        img = val_df.iloc[index, 0]
        label = val_df.iloc[index, 1]
        
        if label != previous_label and previous_label is not None:
            break_index = index
            previous_label = label
            break  
            
        image_path = os.path.join(os.path.join(val_img_folder, file), img + '.jpg')
        val_image = load_image(image_path, transform=data_transforms['val'])
        val_images.append(val_image)
        
        previous_label = label

print(f"Validation Images Count: {len(val_images)}")

label_mapping = {'black': 0,'blue': 1,'brown': 2,'green': 3,'grey': 4,'red': 5,'white': 6,'yellow': 7}

train_labels = [label_mapping[label] for label in train_df.iloc[:, -1].tolist()]
test_labels = [label_mapping[label] for label in test_df.iloc[:, -1].tolist()]
val_labels = [label_mapping[label] for label in val_df.iloc[:, -1].tolist()]

# Define the batch size
batch_size = 32

# Define the datasets
train_dataset = [(img, label) for img, label in zip(train_images, train_labels)]
val_dataset = [(img, label) for img, label in zip(val_images, val_labels)]
test_dataset = [(img, label) for img, label in zip(test_images, test_labels)]

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_classes = len(label_mapping)
print(num_classes)

# Define Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(3, out_channels= 32, kernel_size=3),
            # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
            # output = ((224 - 3 + 2 * 0) / 1) + 1 = 222
            # shape = 32x222x222
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # output = 222 / 2 = 111
            # shape = 32x111x111
            
            nn.Conv2d(32, out_channels= 64, kernel_size=3),
            # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
            # output = ((111 - 3 + 2 * 0) / 1) + 1 = 109
            # shape = 64x109x109
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output = 109 / 2 = 54
            # shape = 64x54x54
            
# =============================================================================
#             nn.Conv2d(64, out_channels= 128, kernel_size= 3),
#             # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
#             # output = ((54 - 3 + 2 * 0) / 1) + 1 = 52
#             # shape = 128x52x52
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size= 2, stride= 2),
#             # output = 52 / 2 = 26
#             # shape = 128x26x26
# =============================================================================
            
# =============================================================================
#             nn.Conv2d(128, out_channels= 256, kernel_size=3, stride=1, padding="same"),
#             # output = ((Input_size - kernal_size + 2 * padding)/ stride) + 1
#             # output = ((26 - 3 + 2 * 0) / 1) + 1 = 24
#             # shape = 256x24x24
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # output = 24 / 2 = 12
#             # shape = 256x12x12
# =============================================================================
            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 54 * 54, 224),  
            nn.ReLU(),
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Instantiate the model
num_classes = len(label_mapping)
model = SimpleCNN(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
count = 0

# Training on training set
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        print(f"{count} Cycle Completed")
        count+=1
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
    count=0
    
# Evaluation on validation set
model.eval()
correct = 0
total = 0
count = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"{count} Cycle Completed")
        count+=1

val_accuracy = correct / total
print(f"Validation Accuracy: {val_accuracy:.2%}")

# Evaluation on test set
model.eval()
all_predictions = []
all_labels = []
count = 0 

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"{count} Cycle Completed")
        count+=1
        
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.2%}")

# Calculate metrics for each class
class_names = list(label_mapping.keys())
class_names
test_f1_scores = f1_score(all_labels, all_predictions, average=None, labels=[0, 1, 2, 3, 4, 5, 6, 7])
test_precision_scores = precision_score(all_labels, all_predictions, average=None, labels=[0, 1, 2, 3, 4, 5, 6, 7])
test_recall_scores = recall_score(all_labels, all_predictions, average=None, labels=[0, 1, 2, 3, 4, 5, 6, 7])

# Calculate metrics
test_f1_score = f1_score(all_labels, all_predictions, average='weighted')
test_precision = precision_score(all_labels, all_predictions, average='weighted')
test_recall = recall_score(all_labels, all_predictions, average='weighted')

print(f"Test F1 Score: {test_f1_score:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")


for i, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"  F1 Score: {test_f1_scores[i]:.4f}")
    print(f"  Precision: {test_precision_scores[i]:.4f}")
    print(f"  Recall: {test_recall_scores[i]:.4f}")

conf_matrix = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(conf_matrix)

# Save the model in H5 format
model_filename = 'car_classification_model.h5'
torch.save(model.state_dict(), model_filename)
print("Model saved as", model_filename)