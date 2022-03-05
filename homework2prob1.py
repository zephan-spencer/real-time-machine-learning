import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torchvision.datasets as datasets

import argparse
import os

from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--modelName', type=str, default="prob1A",
                    help='Name of the desired model')


global args, best_prec1
args = parser.parse_args()

writer = SummaryWriter('runs/' + args.modelName)

class housingDataloader():
    def __init__(self, values, labels, train=True):
        if train:
            self.values = values[:435]
            self.labels = labels[:435]
        else:
            self.values = values[435:]
            self.labels = labels[435:]
    def getLoader(self):
        return zip(self.values, self.labels)


def main():

    from PIL.Image import new
    import numpy as np
    import pandas as pd
    housingDataset = pd.DataFrame(pd.read_csv("data/housing.csv"))
    housingDataset.head()
    trainingVars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    trainingData = housingDataset[trainingVars]
    labels = housingDataset['price']
    t_u = torch.tensor(np.asarray(trainingData))
    t_c = torch.tensor(np.asarray(labels))

    t_un = t_u * .1
    norm = np.linalg.norm(t_c)
    t_cn = t_c/norm

    print(trainingData.shape)
    print(labels.shape)

    trainLoader = housingDataloader(t_un, t_cn, True)
    valLoader = housingDataloader(t_un, t_cn, False)
    # cudnn.benchmark = True
    # # Define the mean and std. devation for the dataset
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # Create the model and send it to the GPU
    if args.modelName == "prob1A":
        model = AModel()
    elif args.modelName == "prob1B":
        model = BModel()

    model

    criterion = nn.MSELoss()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100,150])
    print("Initial validation accuracy:")
    prec1 = validate(valLoader, model, criterion, 1)
    
    for epoch in range(1, args.epochs+1, 1):

        if epoch % 10 == 0:
            # Evaluate the network
            prec1 = validate(valLoader, model, criterion, epoch)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        
        train(trainLoader, model, criterion, optimizer, epoch)

        lr_scheduler.step()

def train(trainLoader, model, criterion, optimizer, epoch):
    # Swap model to train
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    epochAccuracy = []
    epochLoss = []
    
    for input, target in trainLoader.getLoader():
        # target = target.cuda()
        # input_var = input.cuda()
        input_var = input
        target_var = torch.tensor([target.float()])
        # Get network output
        output = model(input_var)
        loss = criterion(output, target_var)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Make the output a float
        output = output.float()
        loss = loss.float()
        # Calculate accuracy and loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        
        epochLoss.append(loss.item())
        epochAccuracy.append(prec1)

    trainLoss = sum(epochLoss)/len(epochLoss)
    trainAccuracy = sum(epochAccuracy)/len(epochAccuracy)
    writer.add_scalar('Housing Training loss', trainLoss, epoch)
    writer.add_scalar('Housing Training Accuracy', trainAccuracy, epoch)
    print('Training Accuracy: {top1.avg:.3f}'.format(top1=top1))

def validate(valLoader, model, criterion, epoch):
    # Swap to evaluation
    model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        top1 = AverageMeter()
        epochAccuracy = []
        epochLoss = []
        for input, target in valLoader.getLoader():
            # target = target.cuda()
            # input_var = input.cuda()
            # target_var = target.cuda()
            input_var = input
            target_var = torch.tensor([target.float()])
            # compute output
            output = model(input_var)

            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))
            
            epochLoss.append(loss.item())
            epochAccuracy.append(prec1)
        
        valLoss = sum(epochLoss)/len(epochLoss)
        valAccuracy = sum(epochAccuracy)/len(epochAccuracy)            
            
        print('Validation Accuracy: {top1.avg:.3f}'.format(top1=top1))
        writer.add_scalar('Housing Validation loss', valLoss, epoch)
        writer.add_scalar('Housing Validation Accuracy', valAccuracy, epoch)

    return top1.avg

class AModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class BModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    
    if output == 0:
        res = 0
    else:
        res = abs(abs((target-output)/output)-1)

    return float(res)

if __name__ == '__main__':
    main()