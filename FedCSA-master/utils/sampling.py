#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_iid_static(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        target_list = list()
        count = i * num_items
        for j in range(num_items):
            target_list.append(count + j)
            # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            # all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = set(target_list)
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    # num_shards, num_imgs = 2000, 30
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set = list()
        # rand_set.append(i * 2)
        # rand_set.append(i * 2 + 1)
        # rand_set = set(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_one(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    # num_shards, num_imgs = 2000, 30
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set = list()
        # rand_set.append(i * 2)
        # rand_set.append(i * 2 + 1)
        # rand_set = set(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''
    # 总类别数
    n_classes = train_labels.max()+1

    # [alpha]*n_clients 如下：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # 得到 62 * 10 的标签分布矩阵，记录每个 client 占有每个类别的比率
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)


    # 记录每个类别对应的样本下标
    # 返回二维数组   np.argwhere()返回不为0的所有数的位置信息[[0]，[3]]，flatten（）则压缩为一维数组[0,3]
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # 定义一个空列表作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        # 将一类数据按概率切分好，np.cumsum()是把后面元素都加上前面的，np.split(c,[])是将从按列表的格式切分
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def mnist_noniid_a(dataset, num_users=1):
    # 初始化客户端数、参数α
    N_CLIENTS = num_users
    DIRICHLET_ALPHA = 0.2

    # train_data = datasets.EMNIST(root=".", split="byclass", download=True, train=True)
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train_data = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # num_cls 为类别总数
    num_cls = len(dataset.classes)

    # 所有数据对应的标签下标
    train_labels = np.array(dataset.targets)

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    # plt.figure(figsize=(20, 3))
    # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
    #          bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    # plt.xticks(np.arange(num_cls), train_data.classes)
    # plt.legend()
    # plt.show()
    return client_idcs


if __name__ == '__main__':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    s = mnist_noniid_a(train_data, 100)

    # print(type(s))
    # print(type(s[0]))
    min = 9999
    result = []
    for i in range(len(s)):
        if len(s[i]) == 0:
            print(i)
        else:
            if len(s[i]) <= min:
                min = len(s[i])
    print("-----------------")
    print(min)
    # print(len(result))


def mnist_noniid_two(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = 2000, 30
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # rand_set = list()
        # rand_set.append(i * 2)
        # rand_set.append(i * 2 + 1)
        # rand_set = set(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST",
                seed=1):
    # assign data to the clients
    seed = 1
    other_group_size = (1 - bias) / (num_labels - 1)  # 0.5/9
    worker_per_group = num_workers / num_labels  # 100/10 = 10

    # assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    server_data = []
    server_label = []

    # compute the labels needed for each class
    # real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
            else:
                raise NotImplementedError
            y = y.as_in_context(ctx)

            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    server_data = np.concatenate(*server_data, dim=0)
    server_label = np.concatenate(*server_label, dim=0)

    each_worker_data = [np.concatenate(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [np.concatenate(*each_worker, dim=0) for each_worker in each_worker_label]

    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
