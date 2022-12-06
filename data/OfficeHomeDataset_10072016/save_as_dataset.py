from PIL import Image
import os
import random

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import pandas as pd



image_size = 32
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

count = {}
test_idxs = []
for category, label in y_tensor:
    key = (category.item(), label.item())
    count[key] = count.get(key, 0) + 1

cur = 0
for key in sorted(count.keys()):
    test_idxs += random.sample(range(cur, cur+count[key]), count[key] // 10)
    cur += count[key]

test_idxs = sorted(test_idxs)
train_idxs = set(range(len(y_tensor))) - set(test_idxs)
train_idxs = sorted(list(train_idxs))

print(len(train_idxs))
print(train_idxs)
print(len(test_idxs))
print(test_idxs)

X_train = X_tensor[train_idxs]
y_train = y_tensor[train_idxs]
X_test = X_tensor[test_idxs]
y_test = y_tensor[test_idxs]
torch.save(X_train, 'X_train.pt')
torch.save(y_train, 'y_train.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_test, 'y_test.pt')
