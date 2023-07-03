import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from train_config import *
from base_utils import cprint
from evaluation.auc_curve import auc_analysis
from evaluation.univariant_KM import paint_uniKM, paint_uniKM_fix_mid, paint_uniKM_fix_trainmid
from preprocess.data_helper import *
from models.HyperG.utils.meter import CIndexMeter
from models.HyperG.hyedge.gather_neighbor import *

data_root, result_root, svs_dir, patch_coors_dir, sampled_vis, \
    patch_ft_dir, split_dir, model_save_path, checkpoint_dir, roc_km_output_save_path = get_config(dataset_name)


def paint(data, name):  # data为多元组的列表即可，每个多元组第一个元素代表n
    x = []
    y_all = []
    for i in range(len(data[0]) - 1):  # number of curves
        y_all.append([])

    for i in range(len(data)):
        x.append(data[i][0])
        for t in range(len(data[0]) - 1):
            y_all[t].append(data[i][t + 1])

    # paint
    # set style
    plt.rc('font', family="Times New Roman")
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fontsize = 25

    plt.xlim(xmax=x[-1] * 1.1, xmin=x[0])
    plt.ylim(ymax=y_all[0][0] + 0.05, ymin=y_all[0][0] - 0.05)

    fontsize = 25

    plt.grid(linestyle='-.', linewidth=1)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    labels = ['Random Ignore', 'Ignore by Uncertainty']
    for i, label in enumerate(labels):
        plt.plot(x, y_all[i], label=label)
    plt.xlabel('Keep Ratio', fontsize=30)
    plt.ylabel('C-Index', fontsize=30)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=20, loc="lower left")
    plt.title(name, fontsize=30)
    save = 'result/improvement_{}.pdf'.format(name)
    plt.tight_layout()
    plt.savefig(save, format='pdf', dpi=300, pad_inches=0)
    print('save image to {}'.format(save))


def load_model_patient_all(split_info, ckpt_path=None, title=None, save_to=None):
    data_root, result_root, svs_dir, patch_coors_dir, sampled_vis, \
    patch_ft_dir, split_dir, model_saveneighbor_distance_graph_dir, checkpoint_dir, roc_km_output_save_path = get_config(dataset_name)

    from train_batch import MODEL
    if ckpt_path is not None:
        model_save_dir = ckpt_path

    with open(split_info, 'rb') as f:  # load split data from file
        data_dict = pickle.load(f)

    dataloaders, dataset_sizes, len_ft = get_dataloaders(data_dict, patch_ft_dir, batch_size=1, filter_DX=filter_DX,
                                                         filter_event=filter_event, filter_tnm=filter_tnm)

    layers = None
    model = MODEL(in_ch=len_ft, n_target=n_target, layers=layers, hiddens=hiddens, dropout=dropout,
                  dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy)
    model = model.to(device)

    model.load_state_dict(torch.load(model_save_dir))
    model.eval()

    train = []  # get the predicted time and the real time
    val = []

    save_ls = []

    c_index = CIndexMeter()

    cnt_phase = 0
    true_phase = 0

    for phase in ['train', 'val']:
        # for fts, st, status, id, dfs in dataloaders[phase]:
        for st, status, id, dfs in dataloaders[phase]:
            # fts = fts.to(device).squeeze(0)
            totalfts = torch.tensor(np.load(osp.join(patch_ft_dir, f'{id[0]}_fts.npy'))).float()

            tot_wsi = int(totalfts.shape[0] / 4000)
            pred = None

            for patient_num in range(tot_wsi):

                fts = totalfts[(4000 * patient_num): (4000 * (patient_num + 1))].to(device)
                st = st.to(device)

                H = neighbor_distance(fts, k_nearest=k_nearest)

                # pred, alpha, feats, feats_mean = model(fts, H)
                tmppred, feats, feats_mean = model(fts, H)

                cnt_phase += 1
                # true_phase += (int(dfs.item()) == (pred > 0.5))

                # print(pred,id)

                nowpred = tmppred[0].item()
                if pred is None:
                    pred = nowpred
                else:
                    pred = pred + nowpred

            pred = pred / tot_wsi
            st = st[0].item()
            status = status[0].item()

            pred_label = int(pred > 0.5)

            train.append([pred, st, status, id]) if phase == 'train' else val.append([pred, st, status])
            save_ls.append({'id': id[0], 'pred': pred, 'st': st, 'status': status})

    json.dump(save_ls, open('{}/pred_result_{}.json'.format(save_to, title), "w"))

    return train, val


def evaluate_roc_km(split_path, checkpoint_path, save_title):
    train_data, val_data = load_model_patient_all(split_path, ckpt_path=checkpoint_path, title=save_title, save_to=roc_km_output_save_path)
    result = {'val': val_data, 'train': train_data}
    auc_analysis(result, title=save_title, save_to=roc_km_output_save_path)
    paint_uniKM(result, time[dataset_name.upper()], title=save_title, save_to=roc_km_output_save_path)


