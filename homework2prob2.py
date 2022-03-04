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
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lrDecayPoints', nargs='+', type=int, default=-1,
                    help='List of network class sizes', required=True)
parser.add_argument('--modelName', type=str, default="prob2A",
                    help='Name of the desired model', required=True)


global args, best_prec1
args = parser.parse_args()

writer = SummaryWriter('runs/' + args.modelName)

def main():

    cudnn.benchmark = True
    # Define the mean and std. devation for the dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Define Data Loaders
    trainLoader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    valLoader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create the model and send it to the GPU
    if args.modelName == "prob2A":
        model = Prob2AModel()
    elif args.modelName == "prob2B":
        model = Prob2BModel()

    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lrDecayPoints)
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
    with tqdm(trainLoader, unit="batch") as tepoch:
        epochAccuracy = []
        epochLoss = []
        for input, target in tepoch:

            tepoch.set_description(f"Epoch {epoch}")
                
            target = target.cuda()
            input_var = input.cuda()
            target_var = target

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
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            tepoch.set_postfix(loss=loss.item(), accuracy=top1.avg)

            epochLoss.append(loss.item())
            epochAccuracy.append(prec1.item())

        trainLoss = sum(epochLoss)/len(epochLoss)
        trainAccuracy = sum(epochAccuracy)/len(epochAccuracy)            
            
        writer.add_scalar('CIFAR-10 Training loss', trainLoss, epoch)
        writer.add_scalar('CIFAR-10 Training Accuracy', trainAccuracy, epoch)

def validate(valLoader, model, criterion, epoch):
    # Swap to evaluation
    model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        top1 = AverageMeter()
        epochAccuracy = []
        epochLoss = []
        for input, target in valLoader:
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            
            epochLoss.append(loss.item())
            epochAccuracy.append(prec1.item())
        
        valLoss = sum(epochLoss)/len(epochLoss)
        valAccuracy = sum(epochAccuracy)/len(epochAccuracy)            
            
        writer.add_scalar('CIFAR-10 Validation loss', valLoss, epoch)
        writer.add_scalar('CIFAR-10 Validation Accuracy', valAccuracy, epoch)
            
        print('Validation Accuracy: {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

class Prob2AModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class Prob2BModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()