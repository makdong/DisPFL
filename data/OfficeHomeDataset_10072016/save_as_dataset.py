from PIL import Image
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import pandas as pd



image_size = 64
converter = lambda x: transforms.Resize((image_size, image_size))(transforms.ToTensor()(Image.open(x)))

image_categories = {}
image_labels = {}
category_num, label_num = 0, 0

X = []
y = []

for category in sorted(os.listdir('./')):
    if os.path.isdir('./' + category):
        print(category)
        if not category in image_categories:
            image_categories[category] = category_num
            category_num += 1

        for label in sorted(os.listdir('./' + category)):
            print(label)
            if os.path.isdir('./' + category + '/' + label):
                if not label in image_labels:
                    image_labels[label] = label_num
                    label_num += 1
                
                for jpg in sorted(os.listdir('./' + category + '/' + label)):
                    img = converter('./' + category + '/' + label + '/' + jpg)
                    X.append(img)
                    y.append(torch.tensor([image_categories[category], image_labels[label]]))

print(category_num)
print(image_categories)
print(label_num)
print(image_labels)

X_tensor = torch.stack(X)
y_tensor = torch.stack(y)
torch.save(X_tensor, 'X.pt')
torch.save(y_tensor, 'y.pt')
