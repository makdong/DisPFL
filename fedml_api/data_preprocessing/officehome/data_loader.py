import logging
import math
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import pdb
from .datasets import OFFICEHOME_truncated


def record_net_data_stats(y_train, net_dataidx_map, logger):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(65):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append ( tmp)
    return net_cls_counts



def _data_transforms_officehome():
    officehome_MEAN = [0.5071, 0.4865, 0.4409]
    officehome_STD = [0.2673, 0.2564, 0.2762]

    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size = (64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(officehome_MEAN, officehome_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size = (64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(officehome_MEAN, officehome_STD),
    ])

    return train_transform, valid_transform


def load_officehome_data(datadir):
    transform_train, transform_test = _data_transforms_officehome()

    officehome_train_ds = OFFICEHOME_truncated(datadir, train=True, download=True, transform=transform_train)
    officehome_test_ds = OFFICEHOME_truncated(datadir, train=False, download=True, transform=transform_test)

    X_train, y_train = officehome_train_ds.data, officehome_train_ds.target
    X_test, y_test = officehome_test_ds.data, officehome_test_ds.target

    return (X_train, y_train, X_test, y_test)

def record_part(y_test, train_cls_counts,test_dataidxs ,logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(65):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        test_cls_counts.append ( tmp)
        logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp) ))
    return


def partition_data( datadir, partition, n_nets, alpha, logger):
    logger.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_officehome_data(datadir)

    y_train_category = [[], [], [], []]
    for category, label in y_train :
        y_train_category[category.item()].append(label)
    for i in range(4):
        y_train_category[i] = torch.tensor(y_train_category[i]).unsqueeze(-1)

    y_train = y_train.flatten()[1::2].unsqueeze(-1)
    y_test = y_test.flatten()[1::2].unsqueeze(-1)

    # print(y_train.shape)
    # print(y_train_category[0].shape) # 0 ~ 14, 94개 left
    # print(y_train_category[1].shape) # 15에 47개, 16 ~ 42, 103개 left
    # print(y_train_category[2].shape) # 43에 38개, 44 ~ 71, 41개 left
    # print(y_train_category[3].shape) # 72에 100개, 73 ~ 99, 44개 left

    # y_test consists of tuple, first is category, second is label
    # target = target.flatten()[1::2][self.dataidxs]
    # y_test, net_dataidx_map, traindata_cls_counts is important.
    # y_test for partitioning, net_dataidx_map for the result of partitioning, traindata_cls_counts for class number

    # traindata_cls_counts done.
    # y_test not changed. done.
    # net_dataidx_map. hell. somebody save me.

    # n_train = X_train.shape[0]
    n_client = n_nets
    n_cls = 65

    n_data_per_clnt = len(y_train) / n_client
    clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
    clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)

    # cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
    # prior_cumsum = np.cumsum(cls_priors, axis=1)

    idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    net_dataidx_map = {}

    for j in range(n_client):
        net_dataidx_map[j] = []

    for j in range(14100):
        index = math.floor(j / 141)
        net_dataidx_map[index].append(j)


    # while np.sum(clnt_data_list) != 0:
    #     curr_clnt = np.random.randint(n_client)
    #     # If current node is full resample a client
    #     # print('Remaining Data: %d' %np.sum(clnt_data_list))
    #     if clnt_data_list[curr_clnt] <= 0:
    #         continue
    #     clnt_data_list[curr_clnt] -= 1
    #     # curr_prior = prior_cumsum[curr_clnt]
    #     while True:
    #         cls_label = np.argmax(np.random.uniform() <= curr_prior)
    #         # Redraw class label if trn_y is out of that class
    #         if cls_amount[cls_label] <= 0:
    #             # cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
    #             continue
    #         cls_amount[cls_label] -= 1
    #         net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])

    #         break



    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger)

    # only use y_test, net_dataidx_map, traindata_cls_counts
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_OFFICEHOME(datadir, train_bs, test_bs, dataidxs=None,test_idxs=None, 
                                cache_train_data=None, cache_train_target=None, cache_test_data=None, cache_test_target=None, logger = None):
    transform_train, transform_test = _data_transforms_officehome()

    dataidxs=np.array(dataidxs)

    logger.info("train_num{}  test_num{}".format(len(dataidxs),len(test_idxs)))
    train_ds = OFFICEHOME_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True, cache_data=cache_train_data, cache_target = cache_train_target)
    test_ds = OFFICEHOME_truncated(datadir, dataidxs=test_idxs, train=False, transform=transform_test, download=True, cache_data=cache_test_data, cache_target = cache_test_target)

    # print(test_ds.target.numpy().shape)
    # train_ds.data.numpy(), train_ds.target.numpy()
    # print(train_ds.numpy().data.shape)
    # print(train_ds.data.numpy().shape)
    # print(test_ds.numpy().target.shape)

    # print(train_ds.target.numpy().shape)
    # data.TensorDataset(test_ds.data.numpy(), test_ds.target.numpy())
    
    # print(train_ds.data.numpy().shape)

    train_dl = data.DataLoader(dataset=data.TensorDataset(train_ds.data, train_ds.target), batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=data.TensorDataset(test_ds.data, test_ds.target), batch_size=test_bs, shuffle=True, drop_last=False)
    return train_dl, test_dl




def load_partition_data_officehome( data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, logger = logger)
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_officehome()
    # cache_train_data_set = dict()

    # cache_train_data_set=CIFAR100(data_dir, train=True, transform=transform_train, download=True)
    # cache_test_data_set = CIFAR100(data_dir, train=False, transform=transform_test, download=True)
    cache_train_data = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/X_train.pt')
    cache_train_target = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/y_train.pt')
    # cache_train_data_set[data] = cache_train_data
    # cache_train_data_set[target] = cache_train_target
    # print(cache_train_data.shape)
    cache_test_data = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/X_test.pt')
    cache_test_target = torch.load('/home/guest-kdh2/DisPFL/data/OfficeHomeDataset_10072016/y_test.pt')
    # cache_test_data_set = (cache_test_data, cache_test_target)

    idx_test = [[] for i in range(65)]
    
    for label in range(65): # number of class
        idx_test[label] = np.where(y_test == label)[0]

    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data) / client_number)

    for client_idx in range(client_number):
        for label in range(65):
            # each has 100 pieces of testing data
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))

        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_OFFICEHOME( data_dir, batch_size, batch_size, dataidxs,test_dataidxs[client_idx] ,cache_train_data=cache_train_data, cache_train_target = cache_train_target, cache_test_data=cache_test_data, cache_test_target = cache_test_target ,logger=logger)
        # print("\n\n\n\n\n")
        # print(train_data_local.data.shape)
        # print(test_data_local.data.shape)
        # print("\n\n\n\n\n")
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        # print(test_data_local_dict[0])
    
    record_part(y_test, traindata_cls_counts, test_dataidxs, logger)
    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
