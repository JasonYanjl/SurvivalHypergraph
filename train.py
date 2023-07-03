import copy
import time
from collections import deque

import torch.optim as optim
from torch.optim import lr_scheduler

from base_utils import *
from evaluate_roc_km import evaluate_roc_km
from models.HyperG.utils import check_dir
from models.HyperG.utils.meter import CIndexMeter
from preprocess.data_helper import *
from train_config import *
from train_utils import automatic_get_K
from models.HyperG.hyedge.gather_neighbor import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

calc_batch_size = 1

data_root, result_root, svs_dir, patch_coors_dir, sampled_vis, \
patch_ft_dir, split_dir, model_save_path, checkpoint_dir, roc_km_output_save_path = get_config(dataset_name)

if model_name.lower() == 'hgrcox':
    from models.hgrcox import *

    MODEL = HGRCox
else:
    exit(0)


def _rank_loss(prediction, T, E):
    current_batch_len = len(prediction)

    theta = prediction.reshape(-1)

    train_time = torch.FloatTensor(T).cuda()
    train_ystatus = torch.FloatTensor(E).cuda()

    cmp_time = train_time.unsqueeze(0).T - train_time.unsqueeze(0)
    cmp_time = (cmp_time > 0) * 1 + (cmp_time == 0) * 0 + (cmp_time < 0) * -1
    calc_status = torch.logical_or(train_ystatus.unsqueeze(0).T, train_ystatus.unsqueeze(0))

    sigmoid_res = torch.sigmoid(theta.unsqueeze(0).T - theta.unsqueeze(0))

    if torch.sum(calc_status) == 0:
        return 0

    loss_nn = torch.sum(calc_status * (0.5 * (1 - cmp_time) * sigmoid_res
                                       + torch.log(1+torch.exp(-sigmoid_res)))) \
              / torch.sum(calc_status)

    return loss_nn


def train(model, criterion, optimizer, scheduler,
          dataloaders, dataset_sizes, n_fold_dataloaders, sensitive, pooling_strategy,
          dropmax_ratio, checkpoint_dir, num_epochs=num_epochs, k=k_nearest, ignore_censored=ignore_censored):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    mini_loss = 1e8
    c_index_queue = {'train': deque(maxlen=5), 'val': deque(maxlen=5)}
    c_best = {'train': 0.0, 'val': 0.0}

    save_epoch_loss = {'train': [], 'val': []}
    save_epoch_cindex = {'train': [], 'val': []}

    las_train_epoch = -1
    for epoch in range(num_epochs):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       '{}_{}_{}_{}.ckpt'.format(model_name, sensitive, pooling_strategy, epoch))
        if os.access(checkpoint_path, os.F_OK):
            try:
                model.load_state_dict(torch.load(checkpoint_path))
                las_train_epoch = epoch
            except:
                pass
    print(las_train_epoch)

    if las_train_epoch != -1:
        try:
            save_info = json.load(open(os.path.join(result_root, 'saveinfo.json'), 'r'))
            save_epoch_loss = save_info['loss']
            save_epoch_cindex = save_info['cindex']
        except:
            save_epoch_loss = {'train': [0] * (las_train_epoch+1), 'val': [0] * (las_train_epoch+1)}
            save_epoch_cindex = {'train': [0] * (las_train_epoch+1), 'val': [0] * (las_train_epoch+1)}

        for epoch in range(las_train_epoch + 1):
            print(f'skip {epoch}')
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()


    if k == 0:
        k = automatic_get_K(model, criterion, optimizer, scheduler, n_fold_dataloaders)

    for epoch in range(las_train_epoch + 1, num_epochs):
        cprint("g", 'Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        cprint("y", "{} - {} - {}- dropmax_ratio: {}".
               format(model_name, sensitive, pooling_strategy, dropmax_ratio))

        loss_epoch = {'train': np.inf, 'val': np.inf}

        now_best = False
        now_best_cindex = False

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set models to training mode
                criterion.train()
            else:
                model.eval()  # Set models to evaluate mode
                criterion.eval()

            running_loss = 0.0
            c_index = CIndexMeter()

            # Iterate over data.
            pred_features = np.zeros((1, 128))
            pred_sts = np.zeros((1,))

            iter = 0
            i_batch = 0
            tot_batch = 0

            # for nowfts, nowst, nowstatus, nowid, nowdfs in dataloaders[phase]:
            for nowst, nowstatus, nowid, nowdfs in dataloaders[phase]:
                tot_batch += 1

                lbl_pred_each = None

                survtime_all = []
                status_all = []
                loss = 0
                now_batch_size = 0

                i_batch += 1
                # if ignore_censored and not all(status):  # ingore censored data
                #     continue


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                for i in range(nowst.shape[0]):
                    if phase == 'train':
                        fts = torch.tensor(np.load(osp.join(patch_ft_dir, f'{nowid[i]}_fts.npy'))).float()

                        fts = fts.to(device)
                        st = nowst[i]
                        st = st.to(device)
                        status = nowstatus[i]

                        with torch.set_grad_enabled(phase == 'train'):
                            if k == -1:
                                H = hyedge_concat([neighbor_distance(fts, 5),
                                                   neighbor_distance(fts, 10), neighbor_distance(fts, 15)])
                            else:
                                H = neighbor_distance(fts, k, is_cross_WSI=True)

                            # Save H

                            pred, feats, feats_mean = model(fts, H)

                            loss += criterion(pred[0], st)
                            c_index.add(pred[0], st)

                            now_batch_size += 1

                            survtime_all.append(st.cpu().numpy())
                            status_all.append(status.numpy())
                            if lbl_pred_each is None:
                                lbl_pred_each = pred
                            else:
                                lbl_pred_each = torch.cat([lbl_pred_each, pred])
                    elif phase == 'val':
                        totalfts = torch.tensor(np.load(osp.join(patch_ft_dir, f'{nowid[i]}_fts.npy'))).float()
                        tot_wsi = int(totalfts.shape[0] / 4000)
                        st = nowst[i]
                        st = st.to(device)
                        status = nowstatus[i]
                        pred = None
                        for patient_num in range(tot_wsi):
                            fts = totalfts[(4000 * patient_num): (4000 * (patient_num + 1))].to(device)
                            with torch.set_grad_enabled(phase == 'train'):
                                if k == -1:
                                    H = hyedge_concat([neighbor_distance(fts, 5),
                                                       neighbor_distance(fts, 10), neighbor_distance(fts, 15)])
                                else:
                                    H = neighbor_distance(fts, k)

                                tmppred, feats, feats_mean = model(fts, H)
                                if pred is None:
                                    pred = tmppred
                                else:
                                    pred = pred + tmppred
                        pred = pred / tot_wsi

                        loss += criterion(pred[0], st)
                        c_index.add(pred, st)

                        now_batch_size += 1

                        survtime_all.append(st.cpu().numpy())
                        status_all.append(status.numpy())
                        if lbl_pred_each is None:
                            lbl_pred_each = pred
                        else:
                            lbl_pred_each = torch.cat([lbl_pred_each, pred])

                # backward + optimize only if in training phase

                survtime_all = np.asarray(survtime_all)
                status_all = np.asarray(status_all)
                loss = loss / now_batch_size
                loss = loss + _rank_loss(lbl_pred_each, survtime_all, status_all)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()

            if phase == 'train':
                scheduler.step()

            pred_features = pred_features[1:]
            pred_sts = pred_sts[1:]

            features_saved_path = os.path.join(checkpoint_dir, 'epoch_features')
            if not os.path.exists(features_saved_path):
                os.makedirs(features_saved_path)
                os.makedirs(os.path.join(features_saved_path, 'train'))
                os.makedirs(os.path.join(features_saved_path, 'val'))

            res_feat_out_path = os.path.join(features_saved_path, phase, f'{epoch}_feature_out.npy')
            res_sts_out_path = os.path.join(features_saved_path, phase, f'{epoch}_sts_out.npy')
            open(res_feat_out_path, 'wb+').close()
            open(res_sts_out_path, 'wb+').close()
            np.save(res_feat_out_path, pred_features)
            np.save(res_sts_out_path, pred_sts)

            epoch_loss = running_loss / tot_batch

            c_index_v = c_index.value()

            if epoch >= 10:
                if phase == 'train' and c_best[phase] < c_index_v:
                    c_best[phase] = c_index_v

                if phase == 'val' and c_best[phase] < c_index_v:
                    c_best[phase] = c_index_v
                    now_best_cindex = True

            # c_best[phase] = max(c_best[phase], c_index_v)

            c_index_queue[phase].append(c_index_v)

            print(f'{phase} Loss: {epoch_loss:.4f}, Cur C-Index: {c_index_v:.4f}, '
                  f'C-mean: {np.array(c_index_queue[phase]).mean(): .4f}, '
                  f'C-Best: {c_best[phase]:.4f}, save to {res_feat_out_path}')

            loss_epoch[phase] = epoch_loss

            save_epoch_loss[phase].append(epoch_loss)
            save_epoch_cindex[phase].append(c_index_v)

            # deep copy the models
            if epoch >= 10:
                if phase == 'val' and epoch_loss < mini_loss:
                    mini_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    now_best = True

        checkpoint_path = os.path.join(checkpoint_dir,
                                       '{}_{}_{}_{}.ckpt'.format(model_name, sensitive, pooling_strategy, epoch))
        cprint("b", "Save checkpoint in {}".format(checkpoint_path))
        checkpoint = copy.deepcopy(model.state_dict())
        torch.save(checkpoint, checkpoint_path)

        if now_best:
            evaluate_roc_km(split_path=split_dir, checkpoint_path=checkpoint_path,
                            save_title=f'{dataset_name}-{model_name}-{sensitive}-{pooling_strategy}-{epoch}-best')

        if now_best_cindex:
            evaluate_roc_km(split_path=split_dir, checkpoint_path=checkpoint_path,
                            save_title=f'{dataset_name}-{model_name}-{sensitive}-{pooling_strategy}-{epoch}-best-cindex')

        save_info = {'loss': save_epoch_loss, 'cindex': save_epoch_cindex}
        json.dump(save_info, open(os.path.join(result_root, 'saveinfo.json'), 'w'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Minimize val Loss: {:4f}'.format(mini_loss))

    # load best models weights
    model.load_state_dict(best_model_wts)
    return model


def main(dataset_name=dataset_name, k_nearest=k_nearest, ignore_censored=ignore_censored):
    show_configs(
        model_name=model_name,
        n_target=n_target,
        dataset_name=dataset_name,
        k_nearest=k_nearest,
        ignore_censored=ignore_censored,
        data_root=data_root,
        result_root=result_root,
        svs_dir=svs_dir,
        patch_coors_dir=patch_coors_dir,
        sampled_vis=sampled_vis,
        patch_ft_dir=patch_ft_dir,
        split_dir=split_dir,
        model_save_path=model_save_path,
        learning_rate=learning_rate
    )

    assert check_dir(data_root, make=False)
    assert check_dir(svs_dir, make=False)
    assert check_dir(patch_ft_dir, make=False)
    check_dir(result_root)
    check_dir(sampled_vis)
    check_dir(patch_coors_dir)

    data_dict = split_train_val_sensored(svs_dir, ratio=0.7, save_split_dir=split_dir, resplit=False,
                                         opti_survival_root=data_root)

    dataloaders, dataset_sizes, len_ft = get_dataloaders(data_dict, patch_ft_dir, batch_size=calc_batch_size, filter_DX=filter_DX,
                                                         filter_event=filter_event, filter_tnm=filter_tnm)

    n_fold_dataloaders = get_n_folds_dataloader(data_dict, patch_ft_dir, filter_DX=filter_DX, filter_event=filter_event,
                                                filter_tnm=filter_tnm)

    model = MODEL(in_ch=len_ft, n_target=n_target, hiddens=hiddens, dropout=dropout,
                  dropmax_ratio=dropmax_ratio, sensitive=sensitive, pooling_strategy=pooling_strategy)
    model = model.to(device)
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    if optimizer == 'adam':
        optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    else:
        optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    cprint('y', 'optimzer: {}, loss_function: {}'.format(optimizer, criterion.__class__.__name__))

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)

    if k_nearest == -1:
        print('*' * 10, 'Concact K_nearest 5, 10, 15 as H', '*' * 10)
    elif k_nearest == 0:
        print('*' * 10, 'Automatic K_nearest by 5 folds', '*' * 10)
    else:
        print('*' * 10, 'K_nearest is: ', k_nearest, '*' * 10)

    model = train(model, criterion, optimizer_ft, exp_lr_scheduler,
                  dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                  n_fold_dataloaders=n_fold_dataloaders, sensitive=sensitive,
                  pooling_strategy=pooling_strategy, dropmax_ratio=dropmax_ratio, checkpoint_dir=checkpoint_dir,
                  num_epochs=num_epochs, k=k_nearest, ignore_censored=ignore_censored)

    print("Save model to: ", model_save_path)
    torch.save(model.cpu().state_dict(), model_save_path)


if __name__ == '__main__':
    """ more config settings in the train_configs.py """
    main()
