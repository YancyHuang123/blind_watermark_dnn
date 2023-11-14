from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


# os.environ['CUDA_VISIBLE_DEVICES'] =GPU
parser = argparse.ArgumentParser(
    description='Pytorch Implement Protection for IP of DNN with CIRAR10')
parser.add_argument('--dataset', default='cifar10', help='mnist|cifar10')
parser.add_argument('--dataroot', default='./data/')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=100) # 100 for cifar10    30 for mnist
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--wm_num', nargs='+', default=[500, 600],  # 1% of train dataset, 500 for cifar10, 600 for mnist
                        help='the number of wm images')
parser.add_argument('--wm_batchsize', type=int, default=20, help='the wm batch size')
parser.add_argument('--lr', nargs='+', default=[0.001, 0.1]) # 0.001 for adam    0.1 for sgd
parser.add_argument('--hyper-parameters',  nargs='+', default=[3, 5, 1, 0.1])
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--seed', default=32, type=int,
                    help='seed for initializing training.')
parser.add_argument('--pretrained', type=bool,
                    default=False, help='use pre-trained model')
parser.add_argument('--wm_train', type=bool, default=True,
                    help='whther to watermark  pre-trained model')
args = parser.parse_args()

cudnn.benchmark = True
if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from models import SSIM 
from models import *
from models.Discriminator import DiscriminatorNet, DiscriminatorNet_mnist
from models.HidingUNet import UnetGenerator, UnetGenerator_mnist

cudnn.benchmark = True

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.deterministic = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load trainset and testset
trainset = torchvision.datasets.CIFAR10(
    root=args.dataroot, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(
    root=args.dataroot, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)
# load the 1% origin sample
trigger_set = torchvision.datasets.CIFAR10(
    root=args.dataroot, train=True, download=True, transform=transform_test)
trigger_loader = torch.utils.data.DataLoader(
    trigger_set, batch_size=args.wm_batchsize, shuffle=False, num_workers=2, drop_last=True)

## load logo
#ieee_logo = torchvision.datasets.ImageFolder(
#    root=args.dataroot+'/IEEE', transform=transform_test)
#ieee_loader = torch.utils.data.DataLoader(ieee_logo, batch_size=1)
#for _, (logo, __) in enumerate(ieee_loader):
#    secret_img = logo.expand(
#        args.wm_batchsize, logo.shape[1], logo.shape[2], logo.shape[3]).cuda()


it=iter(trainloader)
x,y=next(it)
print(x)