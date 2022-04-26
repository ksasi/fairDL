import os
import random
import argparse
import torch
import pickle

import math
import sklearn
import numpy as np
import pandas as pd
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

    pred_df = pd.read_csv(args.predfile)
    pred_df_fold = pred_df.loc[pred_df['fold'] == 1]
    fpr_g, tpr_g, thresholds_g = roc_curve(pred_df_fold['label'], pred_df_fold['LightCNN_29'])
    #print("ROC Vals", flush=True)
    #print(fpr_g, flush=True)
    #print(tpr_g, flush=True)
    #print(thresholds_g, flush=True)
    fpr_gd, fnr_gd, thresholds_gd = det_curve(pred_df_fold['label'], pred_df_fold['LightCNN_29'])
    #print("DET Vals", flush=True)
    #print(fpr_gd, flush=True)
    #print(fnr_gd, flush=True)
    #print(thresholds_gd, flush=True)

    far_val_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    gar = [] # genuine acceptance rate aka verification accuracy which is 1 - FRR
    dict_gar = {}
    for far in far_val_list:
        for item in zip(fpr_gd, fnr_gd):
            if truncate(item[0], len(str(far))-2) == far:
                gar.append(1-item[1])
                dict_gar[far] = 1-item[1]
                break
    print(gar, flush = True)
    with open(args.outdir + '/vacc_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gar, f)

    '''
    #print(fpr_g, flush=True)
    #print(thresholds_g, flush=True)
    threshold_val_g_list = []
    far_val_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    for far in far_val_list:
        for item in zip(fpr_g, thresholds_g):
            #print(item)
            if truncate(item[0], len(str(far))-2) == far:
                #threshold_val_g = item[1]
                threshold_val_g_list.append(item[1])
                break
    print(threshold_val_g_list, flush=True)
    #print("\n", flush=True)
    #index_g = np.where(np.around(fpr_g, 2)==0.01)[0][0]
    #threshold_val_g = thresholds_g[index_g]
    #print(threshold_val_g, flush = True)
    '''
    '''
    pred_df_fold['LightCNN_29_pred'] = [1 if x >= threshold_val_g else 0 for x in pred_df_fold['LightCNN_29']]
    print("Verification Accuracy(Overall) at 0.01 precent FAR :", flush=True)
    print(accuracy(pred_df_fold['label'], pred_df_fold['LightCNN_29_pred']), flush=True)
    '''

    '''
    far_acc_dict = {}
    for thresh in zip(threshold_val_g_list, far_val_list):
        pred_df_fold['LightCNN_29_pred'] = [1 if x >= thresh[0] else 0 for x in pred_df_fold['LightCNN_29']]
        print(pred_df_fold, flush=True)
        ver_acc = accuracy(pred_df_fold['label'], pred_df_fold['LightCNN_29_pred'])
        far_acc_dict[thresh[1]] = ver_acc
    print("Verification Accuracy(Overall) :", far_acc_dict, flush=True)
        #print("Verification Accuracy(Overall) at 0.01 precent FAR :", flush=True)
        #print(accuracy(pred_df_fold['label'], pred_df_fold['LightCNN_29_pred']), flush=True)

    dict_ethnicity = {}
    dict_gender = {}
    dict_attrib = {}
    '''

    #### Disabling all Blocks , except realted to ethnicity for debugging #####
    '''
    for val in ['AF', 'AM', 'BF', 'BM', 'IF', 'IM', 'WF', 'WM']:
        temp_df = pred_df_fold.loc[pred_df_fold['a1'] == val]
        ###fpr, tpr, thresholds = roc_curve(temp_df['label'], temp_df['LightCNN_29'])
        #print(np.around(fpr, 2) , flush=True)
        #print(thresholds , flush=True)
        ###index = np.where(np.around(fpr, 2)==0.1)[0][0]
        ###threshold_val = thresholds[index]
        #print(index, flush=True)
        #print(threshold_val,flush=True)
        ###temp_df['LightCNN_29_pred'] = [1 if x >= threshold_val_g else 0 for x in temp_df['LightCNN_29']]
        #print(temp_df)
        dict_attrib[val] = accuracy(temp_df['label'], temp_df['LightCNN_29_pred'])
        #break
    print("Verification Accuracy for each attribute at 0.01 precent FAR :", flush=True)
    print(dict_attrib , flush=True)
    with open(args.outdir + '/vacc_attrib_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_attrib, f)
    

    for val in ['F', 'M']:
        temp_df = pred_df_fold.loc[pred_df_fold['g1'] == val]
        #fpr, tpr, thresholds = roc_curve(temp_df['label'], temp_df['LightCNN_29'])
        #print(np.around(fpr, 2) , flush=True)
        #print(thresholds , flush=True)
        #index = np.where(np.around(fpr, 2)==0.1)[0][0]
        #threshold_val = thresholds[index]
        #print(index, flush=True)
        #print(threshold_val,flush=True)
        ###temp_df['LightCNN_29_pred'] = [1 if x >= threshold_val_g else 0 for x in temp_df['LightCNN_29']]
        #print(temp_df)
        dict_gender[val] = accuracy(temp_df['label'], temp_df['LightCNN_29_pred'])
        #break
    print("Verification Accuracy for each gender at 0.01 precent FAR :", flush=True)
    print(dict_gender , flush=True)
    with open(args.outdir + '/vacc_gender_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gender, f)

    '''
    dict_ethnicity = {}
    dict_ethnicity_far = {}
    for val in ['A', 'B', 'I', 'W']:
        temp_df = pred_df_fold.loc[pred_df_fold['e1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['LightCNN_29'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
        for far in far_val_list:
            for item in zip(fpr_gd, fnr_gd):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[truncate(item[0], len(str(far))-2)] = 1 - item[1]
                    break
        dict_ethnicity[val] = far_gar_dict
        dict_ethnicity_far[val] = far_gar_dict[0.0001]
    print(dict_ethnicity , flush=True)
    with open(args.outdir + '/vacc_ethnicity_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_ethnicity_far, f)
    with open(args.outdir + '/vacc_ethnicity_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_ethnicity, f)

    dict_gender = {}
    dict_gender_far = {}
    for val in ['F', 'M']:
        temp_df = pred_df_fold.loc[pred_df_fold['g1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['LightCNN_29'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
        for far in far_val_list:
            for item in zip(fpr_gd, fnr_gd):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[truncate(item[0], len(str(far))-2)] = 1 - item[1]
                    break
        dict_gender[val] = far_gar_dict
        dict_gender_far[val] = far_gar_dict[0.0001]
    print(dict_gender , flush=True)
    with open(args.outdir + '/vacc_gender_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gender_far, f)
    with open(args.outdir + '/vacc_gender_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_gender, f)

    dict_attrib = {}
    dict_attrib_far = {}
    for val in ['AF', 'AM', 'BF', 'BM', 'IF', 'IM', 'WF', 'WM']:
        temp_df = pred_df_fold.loc[pred_df_fold['a1'] == val]
        fpr_gd, fnr_gd, thresholds_gd = det_curve(temp_df['label'], temp_df['LightCNN_29'])
        far_gar_dict = {}
        far_val_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
        for far in far_val_list:
            for item in zip(fpr_gd, fnr_gd):
                if truncate(item[0], len(str(far))-2) == far:
                    far_gar_dict[truncate(item[0], len(str(far))-2)] = 1 - item[1]
                    break
        dict_attrib[val] = far_gar_dict
        dict_attrib_far[val] = far_gar_dict[0.0001]

    print(dict_attrib , flush=True)
    with open(args.outdir + '/vacc_attrib_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_attrib_far, f)
    with open(args.outdir + '/vacc_attrib_all_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_attrib, f)


    '''
    for val in ['A', 'B', 'I', 'W']:
        temp_df = pred_df_fold.loc[pred_df_fold['e1'] == val]
        far_acc_dict = {}
        for thresh in zip(threshold_val_g_list, far_val_list):
            temp_df['LightCNN_29_pred'] = [1 if x >= thresh[0] else 0 for x in temp_df['LightCNN_29']]
            ######ver_acc = accuracy(pred_df_fold['label'], pred_df_fold['LightCNN_29_pred'])
            ver_acc = accuracy(temp_df['label'], temp_df['LightCNN_29_pred'])
            far_acc_dict[thresh[1]] = ver_acc
        dict_ethnicity[val] = far_acc_dict
        ###fpr, tpr, thresholds = roc_curve(temp_df['label'], temp_df['LightCNN_29'])
        #print(np.around(fpr, 2) , flush=True)
        #print(thresholds , flush=True)
        ###index = np.where(np.around(fpr, 2)==0.1)[0][0]
        ###threshold_val = thresholds[index]
        #print(index, flush=True)
        #print(threshold_val,flush=True)
        ###temp_df['LightCNN_29_pred'] = [1 if x >= threshold_val_g else 0 for x in temp_df['LightCNN_29']]
        #print(temp_df)
            ##############dict_ethnicity[val] = accuracy(temp_df['label'], temp_df['LightCNN_29_pred'])
        #break
    print("Verification Accuracy for each ethnicity at 0.01 precent FAR :", flush=True)
    print(dict_ethnicity , flush=True)
    with open(args.outdir + '/vacc_ethnicity_' + args.state +'.pkl', 'wb') as f:
        pickle.dump(dict_ethnicity, f)
    '''
    
if __name__ == '__main__':
    main()


