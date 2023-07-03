import copy

from tqdm import tqdm
from models.HyperG.hyedge import neighbor_distance
from models.HyperG.utils.meter import CIndexMeter
from preprocess.data_helper import *
from train_config import *


def automatic_get_K(model, criterion, optimizer, scheduler, n_fold_dataloaders, num_epochs=100, k_range=(5, 15)):
    automatic_log = dict()
    k_res = 5
    best_mean_c = 0.0
    for k in range(*k_range):
        best_model_wts = copy.deepcopy(model.state_dict())
        mini_loss = 1e8

        c_index_queue = {phase: list() for phase in ['train', 'val']}
        c_best = {phase: 0.0 for phase in ['train', 'val']}

        automatic_log[k] = list()
        for n_fold_i, (this_dataloaders, this_dataset_sizes, this_len_ft) in enumerate(n_fold_dataloaders):
            for epoch in range(num_epochs):
                print('Epoch {}/{}/{}'.format(epoch, num_epochs - 1, n_fold_i), 'K_nearest:', k,
                      '\n', '-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set models to training mode
                    else:
                        model.eval()  # Set models to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0
                    c_index = CIndexMeter()

                    # Iterate over data.
                    for fts, st, id in this_dataloaders[phase]:
                        fts = fts.to(device).squeeze(0)
                        st = st.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            if k == -1:
                                from HyperG.hygraph.fusion import hyedge_concat
                                H = hyedge_concat([neighbor_distance(fts, 5),
                                                   neighbor_distance(fts, 10),
                                                   neighbor_distance(fts, 15)])
                            else:
                                H = neighbor_distance(fts, k)
                            # # Save H
                            # np.save(os.path.join(hypergraph_out, '{}.npy'.format(id[0])), H.numpy())

                            pred = model(fts, H)
                            # print(f'pred: {pred.item():.4f}, st: {st.item():.4f}')
                            loss = criterion(pred, st)
                            c_index.add(pred, st)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item()
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / this_dataset_sizes[phase]

                    c_index_v = c_index.value()
                    c_best[phase] = max(c_best[phase], c_index_v)

                    c_index_queue[phase].append(c_index_v)
                    best_mean_c = np.array(c_index_queue[phase]).mean()

                    print(f'{phase} Loss: {epoch_loss:.4f}, Cur C-Index: {c_index_v:.4f}, '
                          f'C-mean: {best_mean_c: .4f}, '
                          f'C-Best: {c_best[phase]:.4f}')

                    # deep copy the models
                    if phase == 'val' and epoch_loss < mini_loss:
                        mini_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()

            automatic_log[k].append({phase: np.array(c_index_queue[phase]).mean() for phase in ['train', 'val']})

            if np.array(c_index_queue['train']).mean() > best_mean_c \
                    or np.array(c_index_queue['train']).mean() > best_mean_c:
                k_res = k

    print('**************************\n' * 5)
    print(automatic_log)
    print('Select k:', k_res, ', the trainset C-index:', best_mean_c)
    return k_res


def extract_features(model, save_features_dir, dataloaders, k=10):
    model.eval()
    if not os.path.exists(save_features_dir):
        os.makedirs(save_features_dir)

    for phase in ['train', 'val']:
        features = None
        features_mean = None
        survival_time = None
        for fts, st, id in tqdm(dataloaders[phase]):
            fts = fts.to(device).squeeze(0)
            st = st.to(device)

            with torch.set_grad_enabled(False):
                H = neighbor_distance(fts, k)
                pred, feats, feats_mean = model(fts, H)
                if features is None:
                    features = feats.cpu()
                    features_mean = feats_mean.cpu()
                else:
                    features = np.vstack((features, feats.cpu()))
                    features_mean = np.vstack((features_mean, feats_mean.cpu()))

                if survival_time is None:
                    survival_time = st.cpu()
                else:
                    survival_time = np.vstack((survival_time, st.cpu()))

        features_path = os.path.join(save_features_dir, '{}_extracted_features.npy'.format(phase))
        survival_time_path = os.path.join(save_features_dir, '{}_survival_time.npy'.format(phase))
        open(features_path, 'wb+').close()
        open(survival_time_path, 'wb+').close()
        np.save(features_path, features)
        np.save(survival_time_path, survival_time)
    print('Finish saving features into {}'.format(save_features_dir))
