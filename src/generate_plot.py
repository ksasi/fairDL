import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='Generates plot of verification accuracy of different attributes for pre-trained vs fine-tuned models')

parser.add_argument("--srcdir", default="/workspace/fairDL/results", type=str, help='path where the results of the dictionaries are stores as files')
parser.add_argument("--outdir", default="/workspace/fairDL/results", type=str, help='path to save the results i.e. plots')

## Ref : https://stackoverflow.com/questions/45968359/plotting-two-dictionaries-in-one-bar-chart-with-matplotlib

def stacked_plot(dict1, dict2, outdir, attrib):
    X = np.arange(len(dict1))
    ax = plt.subplot(111)
    #print(np.array(list(dict1.values())), flush=True)
    ax.bar(X, np.array(list(dict1.values())), width=0.2, color='b', align='center')
    ax.bar(X-0.2, np.array(list(dict2.values())), width=0.2, color='g', align='center')
    ax.legend(('Pre Trained','Fine Tuned'))
    plt.xticks(X, dict1.keys())
    plt.title("Verification Accuracy (GAR)", fontsize=17)
    plt.savefig(outdir + '/stacked_plot_' + str(attrib) + '.pdf', format = 'pdf')
    #plt.show()
    plt.close()

def plot_all(srcdir, outdir, figsize = (8,8)):
    for type in ['attrib','gender','ethnicity']:
        with open(srcdir + '/vacc_' + type + '_Pretrained.pkl', 'rb') as f:
            loaded_dict_p = pickle.load(f)
        with open(srcdir + '/vacc_' + type + '_finetuned.pkl', 'rb') as f:
            loaded_dict_f = pickle.load(f)
        stacked_plot(loaded_dict_p, loaded_dict_f, outdir, type)


def stacked_lineplot(dict, outdir, attrib, traintype):
    for key,val in dict.items():
        plt.plot(np.array(list(val.keys())), np.array(list(val.values())), '.-', label=key)
        plt.xscale('log')
    plt.legend()
    plt.title("Verification Accuracy (GAR) Vs FAR ("+str(attrib)+") - "+str(traintype), fontsize=17)
    plt.savefig(outdir + '/gar_far_plot_' + str(attrib) + '-' + str(traintype) + '.pdf', format = 'pdf')
    plt.close()

def plot_line_all(srcdir, outdir, figsize = (8,8)):
    for type in ['attrib','gender','ethnicity']:
        with open(srcdir + '/vacc_' + type + '_all_Pretrained.pkl', 'rb') as f:
            loaded_dict_p = pickle.load(f)
        with open(srcdir + '/vacc_' + type + '_all_finetuned.pkl', 'rb') as f:
            loaded_dict_f = pickle.load(f)
        stacked_lineplot(loaded_dict_p, outdir, type, 'Pretrained')
        stacked_lineplot(loaded_dict_f, outdir, type, 'finetuned')

def plot_vacc(srcdir, outdir, figsize = (8,8)):
    with open(srcdir + '/vacc_all_' + 'Pretrained.pkl', 'rb') as f:
        loaded_dict_p = pickle.load(f)
    with open(srcdir + '/vacc_all_'  + 'finetuned.pkl', 'rb') as f:
        loaded_dict_f = pickle.load(f)
    key = ['Pretrained', 'finetuned']
    plt.plot(np.array(list(loaded_dict_p.keys())), np.array(list(loaded_dict_p.values())), '.-', label=key[0])
    plt.plot(np.array(list(loaded_dict_f.keys())), np.array(list(loaded_dict_f.values())), '.-', label=key[1])
    plt.xscale('log')
    plt.legend()
    plt.title("Verification Accuracy (GAR) Vs FAR", fontsize=17)
    plt.savefig(outdir + '/gar_far_plot_overall.pdf', format = 'pdf')
    plt.close()


def main():
    global args
    args = parser.parse_args()
    plot_all(srcdir = args.srcdir, outdir = args.outdir, figsize = (10,10))
    plot_line_all(srcdir = args.srcdir, outdir = args.outdir, figsize = (10,10))
    plot_vacc(srcdir = args.srcdir, outdir = args.outdir, figsize = (10,10))

if __name__ == "__main__":
    main()

