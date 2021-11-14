from torch.nn.modules import loss
from torchvision import models
from torchvision.models import resnet
from torch import nn, optim
from torchsummary import summary
from tool import load_data, test, train, Feature_Sg
import torch
import time
import numpy as np
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms as T
from torch import nn
from PIL import Image
from torch.utils import data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# res101 = models.resnet101(pretrained = True)
# for param in res101.parameters():
#     param.requires_grad = False
# numClass = 2
# numFit = res101.fc.in_features
# res101.fc = nn.Linear(numFit, numClass)
# res101.fc = nn.Sequential(nn.Linear(numFit, numClass), nn.Softmax(dim=1))

# vgg19 = models.vgg19_bn(pretrained = True)
# vgg19.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, numClass), nn.Softmax(dim=1))

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False
vgg16.classifier[6] = nn.Linear(4096,2)
# vgg16.features.add_module('0', Feature_Sg())
vgg16 = vgg16.cuda()
# vgg16 = torch.load("vgg16.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg16
batch_size = 1
train_data, test_data = load_data(batch_size, train_root="training_set", test_root="test_set") 
lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# for data, target in tqdm(train_data):
#     print(data.shape)

# print(vgg16)


# from torchsummary import summary
# summary(vgg16.cuda(), (3, 224, 224))
train(model, train_data, test_data, optimizer, device, num_epochs)
test(test_data, model, device)