import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import PneumoniaCNN

transform = transforms.Compose([

    transforms.Resize((224,224)),
    transforms.ToTensor(),


])
test_dataset =datasets.ImageFolder(root='data/chest_xray/test',transform = transform)
test_loader = DataLoader(test_dataset,batch_size=32 , shuffle=False)
model = PneumoniaCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images,labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total+= labels.size(0)
        correct +=(predicted ==labels).sum().item()

        accuracy =100*correct / total 
        print("test accuracy", accuracy)
