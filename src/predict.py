import os
import random
import argparse
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from model import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2, SiameseModel, TripletModel, resnet50_scratch_dag, iresnet18, iresnet34, iresnet50, iresnet100, iresnet200, Resnet50_scratch_dag

import numpy as np
import pandas as pd
import swifter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from collections import Counter

parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--model", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--state", default="Pretrained",type=str,  help='State of the model, possible values are Pretrained & Finetuned')
parser.add_argument("--file", default="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv", type=str, help='bfw datatabel in csv format')
parser.add_argument("--root_path", default="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/", type=str, help='path of bfw dataset')
parser.add_argument("--output_file", default="/workspace/fairDL/results/pred.csv", type=str, help='path of the output file to save the result as csv file')
parser.add_argument("--model_checkpoint", default="/workspace/fairDL/models/LightCNN_29Layers_checkpoint.pth.tar", type=str, help='path of model checkpoint file')



def generate_image_embedding(model, image, transform):
    image = transform(image)
    image = torch.unsqueeze(image, 0).cuda()
    model.cuda().eval()
    embedding = model(image)
    return embedding[1].detach().cpu().numpy()
 

def euclidean_similarity():
    pass


def generate_preds(model, file, root_path, transform, output_path):
    """Function to generate consine similarity of images for face verification"""
    df_table = pd.read_csv(file)
    df_table = df_table.drop(columns=['vgg16', 'resnet50', 'senet50'])
    df_table['img_p1'] = df_table['p1'].swifter.apply(lambda x: cv2.imread(root_path + x))
    df_table['emb_p1'] = df_table['img_p1'].swifter.apply(lambda x: generate_image_embedding(model, x, transform))
    df_table['norm_emb_p1'] = df_table['emb_p1'].swifter.apply(lambda x: x/np.linalg.norm(x))
    df_table['img_p2'] = df_table['p2'].swifter.apply(lambda x: cv2.imread(root_path + x))
    df_table['emb_p2'] = df_table['img_p2'].swifter.apply(lambda x: generate_image_embedding(model, x, transform))
    df_table['norm_emb_p2'] = df_table['emb_p2'].swifter.apply(lambda x: x/np.linalg.norm(x))
    df_table['shape_norm_p1'] = df_table['norm_emb_p1'].swifter.apply(lambda x: x.shape)
    df_table['shape_norm_p2'] = df_table['norm_emb_p2'].swifter.apply(lambda x: x.shape)
    df_table.drop('img_p1', axis=1, inplace=True)
    df_table.drop('img_p2', axis=1, inplace=True)
    print(df_table['emb_p1'])
    print(df_table['norm_emb_p1'])
    df_table.to_csv(path_or_buf=output_path, index=False)
    df_table['VGGFace2'] = df_table[['norm_emb_p1', 'norm_emb_p2']].swifter.apply(lambda x: (cosine_similarity(x["norm_emb_p1"], x["norm_emb_p2"])[0][0]), axis=1)
    df_table.to_csv(path_or_buf=output_path, index=False)
    print(max(df_table['VGGFace2']), flush=True)
    return True

def main():
    global args
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    if args.state == "Pretrained":
        if args.model == "LightCNN_29":
            model = LightCNN_29Layers(num_classes=79077)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "LightCNN_29v2":
            model = LightCNN_29Layers_v2(num_classes=80013)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "LightCNN_9":
            model = LightCNN_9Layers(num_classes=79077)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "VGGFace2":
            model = resnet50_scratch_dag(weights_path = args.model_checkpoint)
        elif args.model == "SiameseModel":
            backbone = LightCNN_29Layers(num_classes=79077)
            backbone = torch.nn.DataParallel(backbone)
            backbone.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            backbone.module.fc2 = torch.nn.Linear(256, 2)
            model = SiameseModel(backbone)
        elif args.model == "ArcFace":
            model = iresnet18(pretrained=True)
            model.load_state_dict(torch.load(args.model_checkpoint))
        else:
            print('Incorrect value for model type \n', flush=True)
    elif args.state == "finetuned":
        if args.model == "LightCNN_29":
            model = LightCNN_29Layers(num_classes=79077)
            model = torch.nn.DataParallel(model)
            model.module.fc2 = torch.nn.Linear(256, 1180)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
        elif args.model == "LightCNN_29v2":
            model = LightCNN_29Layers_v2(num_classes=80013)
            model = torch.nn.DataParallel(model)
            model.module.fc2 = torch.nn.Linear(256, 1180)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
        elif args.model == "LightCNN_9":
            model = LightCNN_9Layers(num_classes=79077)
            model = torch.nn.DataParallel(model)
            model.module.fc2 = torch.nn.Linear(256, 1180)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
        elif args.model == "VGGFace2":
            model = resnet50_scratch_dag(weights_path = None)
            model.load_state_dict(torch.load(args.model_checkpoint, map_location='cpu')['state_dict'])        
        elif args.model == "SiameseModel":
            backbone = LightCNN_29Layers(num_classes=79077)
            backbone = torch.nn.DataParallel(backbone)
            backbone.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            backbone.module.fc2 = torch.nn.Linear(256, 2)
            model = SiameseModel(backbone)
        elif args.model == "ArcFace":
            model = iresnet18(pretrained=True)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
        else:
            print('Incorrect value for model type \n', flush=True)



    if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((144,144)), transforms.CenterCrop((128,128)), transforms.ToTensor()])
    elif args.model == "VGGFace2":
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
    elif args.model == "ArcFace":
        transform = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop((112,112)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    model.eval()
    generate_preds(model=model, file=args.file, root_path=args.root_path, transform=transform, output_path=args.output_file)


if __name__ == '__main__':
    main()