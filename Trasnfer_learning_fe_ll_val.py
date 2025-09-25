#Problem statement: train a cifar10 classifer using pretrained resnet18
#1. Feature extractor+ linear layer
import torch
import torchvision
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os

# transfrom converts the image to numerical format
transform = transforms.Compose([
    transforms.Resize(224),   # Resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

os.environ['HTTP_PROXY']  = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"

batch_size = 4
# loading dataset
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

# ---- split into train + val ----
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=batch_size,
                       shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# loading the pretrained resnet18 model
model= models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False # no training-entire network as feature extractor

model.fc = nn.Linear(model.fc.in_features, 10)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# ---- TRAIN + VALIDATION ----
for epoch in range(5):
    running_loss = 0.0
    model.train()
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    # ----validation accuracy each epoch ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f'Epoch {epoch+1}: Validation Accuracy = {val_acc:.2f}%')

    print('Finished Training')

PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
