#Problem statement: train a cifar10 classifer using pretrained resnet18
#4. Fine tuning + freeze all layers except the last layer
import torch
import torchvision
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
#transfrom converts the image to numerical format
transform = transforms.Compose([
    transforms.Resize(224),       # resize CIFAR10 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
import os

os.environ['HTTP_PROXY']  = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"

batch_size = 32
#loading dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2) #num_of_workers is the threads working in the background .shuffle to avoid bias

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2) #Do not shuffle to maintain consistency

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#loading the pretrained resnet18 model
model= models.resnet18(pretrained=True)

# Collect the last layer
last_layers = [model.fc]

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layer
for layer in last_layers:
    for param in layer.parameters():
        param.requires_grad = True


optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


#train the network
for epoch in range(5):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels = data # get the inputs; data is a list of [inputs, labels]
        optimizer.zero_grad() #zero the parameter gradients
        outputs = model(inputs) #forward
        loss = criterion(outputs, labels)
        #backward
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

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
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
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
















