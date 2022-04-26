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
from model import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2, SiameseModel, TripletModel, resnet50_scratch_dag

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel

parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--model", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--state", default="Pretrained",type=str,  help='State of the model, possible values are Pretrained & Finetuned')
parser.add_argument("--file", default="/workspace/fairDL/data/bfw/bfw-v0.1.5-datatable.csv", type=str, help='bfw datatabel in csv format')
parser.add_argument("--root_path", default="/workspace/fairDL/data/bfw/Users/jrobby/bfw/bfw-cropped-aligned/", type=str, help='path of bfw dataset')
parser.add_argument("--output_file", default="/workspace/fairDL/results/pred.csv", type=str, help='path of the output file to save the result as csv file')
parser.add_argument("--model_checkpoint", default="/workspace/fairDL/models/LightCNN_29Layers_checkpoint.pth.tar", type=str, help='path of model checkpoint file')



def generate_image_embedding(model, image_path, transform):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image)
    image = transform(image)
    image = torch.unsqueeze(image, 0).cuda()
    model.cuda().eval()
    embedding = model(image)
    return embedding[1].detach().cpu().numpy()


def generate_preds(model, file, root_path, transform, output_path):
    """Function to generate consine similarity of images for face verification"""
    df_table = pd.read_csv(file)
    df_table = df_table.drop(columns=['vgg16', 'resnet50', 'senet50'])
    df_table = df_table.loc[df_table['fold'] == 1]
    #df_table['p1'] = df_table.apply(lambda x : root_path + x['p1'], axis=1)
    #df_table['p2'] = df_table.apply(lambda x : root_path + x['p2'], axis=1)
    #df_table['embed1'] = df_table.apply(lambda x : generate_image_embedding(model, x['p1'], transform), axis=1)
    #df_table['embed2'] = df_table.apply(lambda x : generate_image_embedding(model, x['p2'], transform), axis=1)
    #minx = -1 
    #maxx = 1
    #df_table['LightCNN_29'] = df_table.apply(lambda x: (cosine_similarity(generate_image_embedding(model, root_path + x['p1'], transform), generate_image_embedding(model, root_path + x['p2'], transform))[0][0]- minx)/(maxx-minx), axis=1)
    ##minx = 0 
    ##maxx = 1
    ##df_table['LightCNN_29'] = df_table.apply(lambda x: (cosine_similarity(generate_image_embedding(model, root_path + x['p1'], transform), generate_image_embedding(model, root_path + x['p2'], transform))[0][0]- minx)/(maxx-minx), axis=1)
    ##df_table['LightCNN_29'] = df_table.apply(lambda x: 0 if x['LightCNN_29'] <0 else x['LightCNN_29'], axis=1)
    #df_table['LightCNN_29'] = df_table.apply(lambda x: 1/(1 + euclidean_distances(generate_image_embedding(model, root_path + x['p1'], transform)/np. linalg. norm(generate_image_embedding(model, root_path + x['p1'], transform)), generate_image_embedding(model, root_path + x['p2'], transform)/np. linalg. norm(generate_image_embedding(model, root_path + x['p2'], transform)))[0][0]), axis=1)
    ####df_table['LightCNN_29'] = df_table.apply(lambda x: 1/(1 + euclidean_distances(generate_image_embedding(model, root_path + x['p1'], transform), generate_image_embedding(model, root_path + x['p2'], transform))[0][0]), axis=1)
    df_table['LightCNN_29'] = df_table.apply(lambda x: 1 - ((euclidean_distances(generate_image_embedding(model, root_path + x['p1'], transform)/np. linalg. norm(generate_image_embedding(model, root_path + x['p1'], transform)), generate_image_embedding(model, root_path + x['p2'], transform)/np. linalg. norm(generate_image_embedding(model, root_path + x['p2'], transform)))[0][0])/2), axis=1)
    df_table.to_csv(path_or_buf=output_path, index=False)
    print(max(df_table['LightCNN_29']), flush=True)
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
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "LightCNN_29v2":
            model = LightCNN_29Layers_v2(num_classes=80013)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "LightCNN_9":
            model = LightCNN_9Layers(num_classes=79077)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
            model.module.fc2 = torch.nn.Linear(256, 1180)
        elif args.model == "VGGFace2":
            #model = resnet50_pt(weights_path = '/workspace/MTP/face-recognition/models/resnet50_scratch_weight.pkl', num_classes=8631, include_top=True)
            model = resnet50_scratch_dag(weights_path = args.model_checkpoint)
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
            #model.fc = torch.nn.Linear(2048, args.num_classes)
        elif args.model == "SiameseModel":
            backbone = LightCNN_29Layers(num_classes=79077)
            backbone = torch.nn.DataParallel(backbone)
            backbone.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            #set_parameter_requires_grad(backbone, feature_extracting=True, num_layers=10)
            backbone.module.fc2 = torch.nn.Linear(256, 2)
            model = SiameseModel(backbone)
        elif args.model == "ArcFace":
            model = iresnet18(pretrained=True)
            model.load_state_dict(torch.load(args.model_checkpoint))
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
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
            #model = resnet50_pt(weights_path = '/workspace/MTP/face-recognition/models/resnet50_scratch_weight.pkl', num_classes=8631, include_top=True)
            model = resnet50_scratch_dag(weights_path = args.model_checkpoint)
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
            #model.fc = torch.nn.Linear(2048, args.num_classes)
        elif args.model == "SiameseModel":
            backbone = LightCNN_29Layers(num_classes=79077)
            backbone = torch.nn.DataParallel(backbone)
            backbone.load_state_dict(torch.load(args.model_checkpoint)['state_dict'])
            #set_parameter_requires_grad(backbone, feature_extracting=True, num_layers=10)
            backbone.module.fc2 = torch.nn.Linear(256, 2)
            model = SiameseModel(backbone)
        elif args.model == "ArcFace":
            model = iresnet18(pretrained=True)
            model.load_state_dict(torch.load(args.model_checkpoint))
            #set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        else:
            print('Incorrect value for model type \n', flush=True)



    if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
        transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(128), transforms.ToTensor()])
    elif args.model == "VGGFace2":
        #transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(128), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
    elif args.model == "ArcFace":
        transform = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(112), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    model.eval()
    generate_preds(model=model, file=args.file, root_path=args.root_path, transform=transform, output_path=args.output_file)


if __name__ == '__main__':
    main()