import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

from model import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2, SiameseModel, TripletModel, resnet50_scratch_dag, iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

from train import train_epoch, val_epoch, train_model, train_val_split

from utils import generate_embeddings, generate_embeddings_v2, cosine_similarity_matrix, get_cmc_scores, plot_cmc_curve, pairselector, detectalign

from pytorch_metric_learning import losses, miners, distances, reducers, testers

import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--model", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--dataset", default="CMU", type=str, help='Dataset to be used for training(default is LFW)')
parser.add_argument("--epochs", default=50, type=int, help='epochs for training (default value is 50)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size for training (default value is 128)')
parser.add_argument("--learning_rate", default=0.01, type=float, help='initial learning rate for training (default value is 0.01)')
parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default value is 0.9)')
parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay (default value is 1e-4)')
parser.add_argument("--arch", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--num_classes", default= 10,type=int, help='number of classes (default value is 10)')
parser.add_argument("--save_path", default="", type=str, help='path to save the checkpoint file(default is None)')
parser.add_argument("--val_list", default="", type=str, help='path to validation list(default is None)')
parser.add_argument("--train_list", default="", type=str, help='path to training list(default is None)')



def set_parameter_requires_grad(model, feature_extracting, num_layers):
    if feature_extracting:
        num = 0
        for param in model.parameters():
            num = num + 1
            if num > len(list(model.parameters())) - num_layers:
               param.requires_grad = True
            else:
               param.requires_grad = False

def set_parameter_requires_grad_no_head(model, feature_extracting, num_layers):
    if feature_extracting:
        num = 0
        for param in model.parameters():
            num = num + 1
            if num < len(list(model.parameters())) - num_layers:
               param.requires_grad = True
            else:
               param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if args.model == "LightCNN_29":
        model = LightCNN_29Layers(num_classes=79077)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('/workspace/fairDL/models/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        #model.module.fc2 = torch.nn.Linear(256, args.num_classes)
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        #unfreeze_all(model)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "LightCNN_29v2":
        model = LightCNN_29Layers_v2(num_classes=80013)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('/workspace/fairDL/models/LightCNN_29Layers_V2_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "LightCNN_9":
        model = LightCNN_9Layers(num_classes=79077)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('/workspace/fairDL/models/LightCNN_9Layers_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "VGGFace2":
        model = resnet50_scratch_dag(weights_path = '/workspace/fairDL/models/resnet50_scratch_dag.pth') # Original weights of VGGFace2
        freeze_all(model)
        model.conv5_3_1x1_increase.requires_grad_(True)
        model.conv5_3_3x3.requires_grad_(True)
    elif args.model == "SiameseModel":
        backbone = LightCNN_29Layers(num_classes=79077)
        backbone = torch.nn.DataParallel(backbone)
        backbone.load_state_dict(torch.load('/workspace/fairDL/models/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(backbone, feature_extracting=True, num_layers=10)
        backbone.module.fc2 = torch.nn.Linear(256, 2)
        model = SiameseModel(backbone)
    elif args.model == "ArcFace":
        model = iresnet18(pretrained=True)
        model.load_state_dict(torch.load('/workspace/fairDL/models/ms1mv3_arcface_r18_fp16/backbone.pth'))
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
    else:
        print('Incorrect value for model type \n', flush=True)


    if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
        train_transforms = transforms.Compose([transforms.Resize((144,144)), transforms.RandomCrop((128,128)), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        val_transforms = transforms.Compose([transforms.Resize((144,144)), transforms.RandomCrop((128,128)), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    elif args.model == "VGGFace2":
        train_transforms = transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop((224,224)), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
        val_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
    elif args.model == "ArcFace":
        train_transforms = transforms.Compose([transforms.CenterCrop((112,112)), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        val_transforms = transforms.Compose([transforms.CenterCrop((112,112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        pass
    
    if args.dataset == "CMU":
        train_dataset = torchvision.datasets.ImageFolder('/workspace/fairDL/data/MultiPie51train/', transform=train_transforms)
        val_dataset = torchvision.datasets.ImageFolder('/workspace/fairDL/data/MultiPie51test/', transform=val_transforms)
    elif args.dataset == "Synth":
        train_dataset = torchvision.datasets.ImageFolder('/workspace/fairDL/data/synthface_processed/', transform=train_transforms)
        val_dataset = torchvision.datasets.ImageFolder('/workspace/fairDL/data/synthfaceval_processed/', transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True) # Classes may still be imbalaced during forward pass (Ref: https://github.com/adambielski/siamese-triplet) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    criterion = losses.ArcFaceLoss(num_classes=2000, embedding_size=2048, margin=35, scale=64)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_optimizer  = torch.optim.SGD(criterion.parameters(), lr = 0.01, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, criterion_optimizer, optimizer, lr_scheduler, args.epochs, args.save_path, args.arch)

if __name__ == '__main__':
    main()
