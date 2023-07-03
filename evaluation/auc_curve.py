import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import copy

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from sklearn.metrics import roc_auc_score

import random

def auc_analysis(data, title, save_to):
    def data_convert(data):
        pred = [p[0] for p in data]
        label = [p[1] for p in data]
        sorted_label = copy.deepcopy(label)
        sorted_label.sort()
        mid = sorted_label[int(len(label) / 2)]
        sorted_pred = copy.deepcopy(pred)
        sorted_pred.sort()
        pred_mid = sorted_pred[int(len(pred) / 2)]
        binary_label = [1 if t >= mid else 0 for t in label]
        return binary_label, pred

    def paint_multi_auc(labels, predicts, name, n_classes, title):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels[i], predicts[i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        lw = 2
        plt.figure()

        # set style
        plt.rc('font', family="Times New Roman")
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        fontsize = 22

        colors = cycle(['r', 'b', 'g', 'k', 'c', 'm', 'teal', 'gold', 'yellowgreen'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC {0} (area = {1:0.2f})'''.format(name[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.title('ROC Curve of {}'.format(title.split('-')[0]), fontsize=25)

        plt.grid(linestyle='-.', linewidth=1)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=15, loc="lower right")
        plt.tight_layout()
        plt.savefig(fname='{}/auc_result_{}.png'.format(save_to, title), format='png', dpi=300, pad_inches=0)
        plt.close()

        return roc_auc

    keys = data.keys()
    name = []
    y_labels = []
    y_preds = []
    for key in keys:
        name.append(key)
        binary_label, pred = data_convert(data[key])
        y_labels.append(binary_label)
        y_preds.append(pred)
    roc_auc = paint_multi_auc(y_labels, y_preds, name, len(name), title=title)

    return roc_auc
