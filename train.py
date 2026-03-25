import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import PneumoniaCNN
import torch.optim as optim
import torch.nn as nn


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

model = PneumoniaCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("model loaded successfully")


print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))
print("Number of test images:", len(test_dataset))


epochs = 5 

for epoch in range(epochs) :

    total = 0 
    correct = 0
    running_loss = 0.0


for images , labels in train_loader : 

    optimizer.zero_grad() 
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()            
    optimizer.step()    

running_loss += loss.item()
_, predicted = torch.max(outputs, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")


print("training completed")

