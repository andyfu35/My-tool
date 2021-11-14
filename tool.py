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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(model, train_data, valid_data, optimizer, device, num_epochs):
    model = model.to(device)
    print("training on ", device)
    valid_loss_min = np.Inf
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        print('running epoch: {}'.format(epoch+1))
        model.train()
        for data, target in tqdm(train_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            l = loss(output, target)
            l.backward()
            optimizer.step()
            train_loss += l.item()*data.size(0)

        model.eval()
        for data, target in tqdm(valid_data):
            data, target = data.to(device), target.to(device)
            output = model(data)
            l = loss(output, target)
            valid_loss += l.item()*data.size(0)
        
        train_loss = train_loss/len(train_data.dataset)
        valid_loss = valid_loss/len(valid_data.dataset)
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model, 'vgg16.pth')
            valid_loss_min = valid_loss


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n



def load_data(batch_size, train_root, test_root):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize([224,224]),
        T.ToTensor(),
        normalize
    ])
    test_transform = T.Compose([
        T.Resize([224,224]),
        T.ToTensor(),                                                                            
        normalize
    ])
    train_dataset = ImageFolder(train_root, transform=train_transform)
    test_dataset = ImageFolder(test_root, transform=test_transform)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_data, test_data

def test(test_data, model, device):
    test_loss = 0.
    correct = 0.
    total = 0.
    loss = torch.nn.CrossEntropyLoss()
    model.eval()

    for ii, (data, target) in enumerate(test_data):
        data, target = data.to(device), target.to(device)
        output = model(data)
        l = loss(output, target)
        test_loss = test_loss + ((1 / (ii + 1)) * (l.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))


class Feature_Sg(torch.nn.Module):
    def __init__(self):
        super(Feature_Sg, self).__init__()
    def forward(self, data):
        max = torch.max(data)
        data = data / max
        data = torch.pow(data, 2)
        data = data * 255
        data = torch.round(data)
        return data
