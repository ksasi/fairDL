import os
import random
import argparse
import torch
import pickle

import math
import sklearn
import numpy as np
import pandas as pd
#import cudf
from utils import precision,recall,f1_score,confusion_matrix, roc_curve, det_curve, accuracy



parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--state", default="Pretrained",type=str,  help='State of the model, possible values are Pretrained & Finetuned')
parser.add_argument("--predfile", default="/workspace/fairDL/results/outputs.csv", type=str, help='csv file containing predictions along with other columns')
parser.add_argument("--outdir", default="/workspace/fairDL/results", type=str, help='path to save the results')

# Ref : https://stackoverflow.com/questions/29246455/python-setting-decimal-place-range-without-rounding
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def main():
    global args
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    print("Before DF")
    pred_df = pd.read_csv(args.predfile)
    pred_df_fold = pred_df
    fpr_g, tpr_g, thresholds_g = roc_curve(pred_df_fold['label'], pred_df_fold['VGGFace2'])
    fpr_gd, fnr_gd, thresholds_gd = det_curve(pred_df_fold['label'], pred_df_fold['VGGFace2'])

    print("Before Loop")
    far_val_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gar = [] # genuine acceptance rate aka verification accuracy which is 1 - FRR
    dict_gar = {}
    print(fpr_g)
    print(tpr_g)
    print(fpr_gd)
    print(fnr_gd)

    for far in far_val_list:
        for item in zip(fpr_g, tpr_g):
            if truncate(item[0], len(str(far))-2) == far:
                gar.append(item[1])
                dict_gar[far] = item[1]
                break

    print(gar, flush = True)
    with open(args.outdir + '/vacc_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gar, f)
    print("After Dump")

    dict_ethnicity = {}
    dict_ethnicity_far = {}
    for val in ['A', 'B', 'I', 'W']:
        temp_df = pred_df_fold.loc[pred_df_fold['e1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['VGGFace2'])
        fpr_g, tpr_g, thresholds_g = roc_curve(temp_df['label'], temp_df['VGGFace2'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for far in far_val_list:
            for item in zip(fpr_g, tpr_g):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[far] = item[1]
                    break
        dict_ethnicity[val] = far_gar_dict
        dict_ethnicity_far[val] = far_gar_dict[0.01]
    print(dict_ethnicity , flush=True)
    with open(args.outdir + '/vacc_ethnicity_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_ethnicity_far, f)
    with open(args.outdir + '/vacc_ethnicity_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_ethnicity, f)

    dict_gender = {}
    dict_gender_far = {}
    for val in ['F', 'M']:
        temp_df = pred_df_fold.loc[pred_df_fold['g1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['VGGFace2'])
        fpr_g, tpr_g, thresholds_g = roc_curve(temp_df['label'], temp_df['VGGFace2'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for far in far_val_list:
            for item in zip(fpr_g, tpr_g):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[far] = item[1]
                    break
        dict_gender[val] = far_gar_dict
        dict_gender_far[val] = far_gar_dict[0.01]
    print(dict_gender , flush=True)
    with open(args.outdir + '/vacc_gender_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gender_far, f)
    with open(args.outdir + '/vacc_gender_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gender, f)

    dict_attrib = {}
    dict_attrib_far = {}
    for val in ['AF', 'AM', 'BF', 'BM', 'IF', 'IM', 'WF', 'WM']:
        temp_df = pred_df_fold.loc[pred_df_fold['a1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['VGGFace2'])
        fpr_g, tpr_g, thresholds_g = roc_curve(temp_df['label'], temp_df['VGGFace2'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for far in far_val_list:
            for item in zip(fpr_g, tpr_g):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[far] = item[1]
                    break
        dict_attrib[val] = far_gar_dict
        dict_attrib_far[val] = far_gar_dict[0.01]
    print(dict_attrib , flush=True)
    with open(args.outdir + '/vacc_attrib_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_attrib_far, f)
    with open(args.outdir + '/vacc_attrib_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_attrib, f)

if __name__ == '__main__':
    main()


