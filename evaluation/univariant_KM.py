from lifelines.utils import median_survival_times
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dd
from matplotlib import pyplot as plt
import matplotlib
from lifelines.statistics import logrank_test
import copy
import pandas as pd


def data_convert(data):
    pred = [p[0] for p in data]
    label = [p[1] for p in data]
    status = [p[2] for p in data]
    sorted_label = copy.deepcopy(pred)
    sorted_label.sort()
    mid = sorted_label[int(len(pred) / 2)]
    binary_pred = [1 if t > mid else 0 for t in pred]
    print()
    return binary_pred, label, status


def paint_uniKM(data, time_max, title, save_to):
    # set style
    plt.rc('font', family="Times New Roman")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fontsize = 25

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    # get data of curve
    p_record = list()
    for i, phase in enumerate(data.keys()):
        data_train = data[phase]
        binary_pred, ground_time_ratio, status = data_convert(data_train)
        ground_time = [gt * time_max for gt in ground_time_ratio]
        data_df = pd.DataFrame({'binary_pred': binary_pred, 'ground_time': ground_time, 'status': status})

        T = data_df['ground_time']
        E = data_df['status']
        binary_pred = data_df['binary_pred']

        kmf = KaplanMeierFitter()
        low_risk = (binary_pred == 1)

        ax = axes[i]
        kmf.fit(T[low_risk], event_observed=E[low_risk], label="low risk")
        kmf.plot(ax=ax, color='g', fontsize=fontsize)

        kmf.fit(T[~low_risk], event_observed=E[~low_risk], label="high risk")
        kmf.plot(ax=ax, color='r', fontsize=fontsize)

        plt.ylim(0, 1)

        ax.grid(linestyle='-.', linewidth=1)
        # ax = plt.gca()

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        # ax.legend(loc=1, fontsize=35)
        ax.legend(fontsize = 25, loc = "lower left")

        log_test = logrank_test(T[low_risk], T[~low_risk], E[low_risk], E[~low_risk], alpha=.99)
        p_value = log_test.p_value
        p_record.append(p_value)

        ax.set_ylabel('Probability', fontsize=40)
        ax.set_xlabel('{}: p={:.3e}'.format(phase, p_value), fontsize=40)

    plt.suptitle(title.split('-')[0], fontsize=40)
    # plt.title(title, fontsize=40)
    plt.tight_layout()
    plt.savefig(fname='{}/uniKM_result_{}.png'.format(save_to, title), format='png', dpi=300, pad_inches=0)
    plt.close()

    return p_record


def data_convert_fix_mid(data, fix_mid):
    pred = [p[0] for p in data]
    label = [p[1] for p in data]
    status = [p[2] for p in data]
    sorted_label = copy.deepcopy(pred)
    sorted_label.sort()
    mid = sorted_label[int(len(pred) / 2)]
    binary_pred = [1 if t > fix_mid else 0 for t in pred]
    print()
    return binary_pred, label, status


def paint_uniKM_fix_mid(data, time_max, title, save_to, fix_mid):
    # set style
    plt.rc('font', family="Times New Roman")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fontsize = 25

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    # get data of curve
    p_record = list()
    for i, phase in enumerate(data.keys()):
        data_train = data[phase]
        binary_pred, ground_time_ratio, status = data_convert_fix_mid(data_train, fix_mid)
        ground_time = [gt * time_max for gt in ground_time_ratio]
        data_df = pd.DataFrame({'binary_pred': binary_pred, 'ground_time': ground_time, 'status': status})

        T = data_df['ground_time']
        E = data_df['status']
        binary_pred = data_df['binary_pred']

        kmf = KaplanMeierFitter()
        low_risk = (binary_pred == 1)

        ax = axes[i]
        kmf.fit(T[low_risk], event_observed=E[low_risk], label="low risk")
        kmf.plot(ax=ax, color='g', fontsize=fontsize)

        kmf.fit(T[~low_risk], event_observed=E[~low_risk], label="high risk")
        kmf.plot(ax=ax, color='r', fontsize=fontsize)

        plt.ylim(0, 1)

        ax.grid(linestyle='-.', linewidth=1)
        # ax = plt.gca()

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        # ax.legend(loc=1, fontsize=35)
        ax.legend(fontsize = 25, loc = "lower left")

        log_test = logrank_test(T[low_risk], T[~low_risk], E[low_risk], E[~low_risk], alpha=.99)
        p_value = log_test.p_value
        p_record.append(p_value)

        ax.set_ylabel('Probability', fontsize=40)
        ax.set_xlabel('{}: p={:.3e}'.format(phase, p_value), fontsize=40)

    plt.suptitle(title.split('-')[0], fontsize=40)
    # plt.title(title, fontsize=40)
    plt.tight_layout()
    plt.savefig(fname='{}/uniKM_result_{}.png'.format(save_to, title), format='png', dpi=300, pad_inches=0)
    plt.close()

    return p_record

def data_convert_fix_trainmid(data, train_mid):
    pred = [p[0] for p in data]
    label = [p[1] for p in data]
    status = [p[2] for p in data]
    sorted_label = copy.deepcopy(pred)
    sorted_label.sort()
    mid = sorted_label[int(len(pred) / 2)]
    binary_pred = [1 if t > train_mid else 0 for t in pred]
    print()
    return binary_pred, label, status


def get_train_mid(data):
    pred = [p[0] for p in data]
    sorted_label = copy.deepcopy(pred)
    sorted_label.sort()
    mid = sorted_label[int(len(pred) / 2)]
    return mid


def paint_uniKM_fix_trainmid(data, time_max, title, save_to):
    # set style
    plt.rc('font', family="Times New Roman")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fontsize = 25

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    train_mid = get_train_mid(data['train'])
    # get data of curve
    p_record = list()
    for i, phase in enumerate(data.keys()):
        data_train = data[phase]
        binary_pred, ground_time_ratio, status = data_convert_fix_trainmid(data_train, train_mid)
        ground_time = [gt * time_max for gt in ground_time_ratio]
        data_df = pd.DataFrame({'binary_pred': binary_pred, 'ground_time': ground_time, 'status': status})

        T = data_df['ground_time']
        E = data_df['status']
        binary_pred = data_df['binary_pred']

        kmf = KaplanMeierFitter()
        low_risk = (binary_pred == 1)

        ax = axes[i]
        kmf.fit(T[low_risk], event_observed=E[low_risk], label="low risk")
        kmf.plot(ax=ax, color='g', fontsize=fontsize)

        kmf.fit(T[~low_risk], event_observed=E[~low_risk], label="high risk")
        kmf.plot(ax=ax, color='r', fontsize=fontsize)

        plt.ylim(0, 1)

        ax.grid(linestyle='-.', linewidth=1)
        # ax = plt.gca()

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)

        # ax.legend(loc=1, fontsize=35)
        ax.legend(fontsize = 25, loc = "lower left")

        log_test = logrank_test(T[low_risk], T[~low_risk], E[low_risk], E[~low_risk], alpha=.99)
        p_value = log_test.p_value
        p_record.append(p_value)

        ax.set_ylabel('Probability', fontsize=40)
        ax.set_xlabel('{}: p={:.3e}'.format(phase, p_value), fontsize=40)

    plt.suptitle(title.split('-')[0], fontsize=40)
    # plt.title(title, fontsize=40)
    plt.tight_layout()
    plt.savefig(fname='{}/uniKM_result_{}.png'.format(save_to, title), format='png', dpi=300, pad_inches=0)
    plt.close()

    return p_record