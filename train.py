import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.ToTensor() ,

])

train_dataset = datasets.ImageFolder(root='data/chest_xray/train', transform=transform)
val_dataset = datasets.ImageFolder(root='data/chest_xray/val', transform=transform)
test_dataset = datasets.ImageFolder(root='data/chest_xray/test', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of test images:", len(test_dataset))

