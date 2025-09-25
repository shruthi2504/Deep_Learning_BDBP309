#Problem statement: train a cifar10 classifer using pretrained resnet18
#3. Feature extractor + SVM
import numpy as np
import torch
import torchvision
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torchvision.transforms as transforms
#transform converts the image to numerical format
transform = transforms.Compose([
    transforms.Resize(224),            # resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

import os

os.environ['HTTP_PROXY']  = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"
os.environ['HTTPS_PROXY'] = "http://245hsbd014%40ibab.ac.in:ibabstudent@proxy.ibab.ac.in:3128/"

batch_size = 4
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


for param in model.parameters():
    param.requires_grad = False #no training-entire network as feature extractor

# model.fc = nn.Identity() #keep only the vector-remove the classifier
model.eval()
#extract features from last fc layer after removing classifier layer
#use those features to train a svm classifier
print(model)
return_nodes = {'fc': 'features'}  # we want output of avgpool
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
print(type(feature_extractor))



def extract_features(loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            features = feature_extractor(images)['features']  # [batch_size=4,1000]
            all_features.append(features.numpy())  # convert tensor to numpy
            all_labels.append(labels.numpy())
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        return X, y


X_train, y_train = extract_features(trainloader)
X_test, y_test = extract_features(testloader)

print("Train features:", X_train.shape)  #  (50000, 1000)
print("Test features:", X_test.shape)  # (10000, 1000)

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Convert features to DataFrame
df_train = pd.DataFrame(X_train)
df_train['label'] = y_train

df_test = pd.DataFrame(X_test)
df_test['label'] = y_test

print(df_train.head())   # preview first rows
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# Split features and labels
Xtr, ytr = df_train.drop(columns=['label']), df_train['label']
Xte, yte = df_test.drop(columns=['label']), df_test['label']

# Train an SVM
svm = SVC(kernel='linear', C=1.0)   # linear kernel, can try 'rbf'
svm.fit(Xtr, ytr)

# Predictions
y_pred = svm.predict(Xte)

# Accuracy
acc = accuracy_score(yte, y_pred)
print("Test Accuracy:", acc)

# Classification report
print(classification_report(yte, y_pred, target_names=classes))
