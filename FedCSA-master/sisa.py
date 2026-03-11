#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
from torchvision import datasets, transforms
import torch
import csv
import time


from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_iid_static, mnist_noniid_two, mnist_noniid_a, cifar_noniid
from utils.div_class import difference_model, difference_model_middle, dsds, difference_model_middle_ma, \
    difference_model_middle_ou, difference_model_middle_k, difference_model_middle_midu, dsds_fiveoo
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdate_Dou
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(torch.cuda.is_available())
    print(args.device)
    print(args.dataset)

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    # copy weights
    w_glob = copy.deepcopy(net_glob.state_dict())

    # training
    loss_train = []
    loss_train_sds = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        # w_locals2 = [w_glob for i in range(args.num_users)]
    # for iter in range(args.epochs):
    loss_locals = []
    if not args.all_clients:
        w_locals = []
        w_locals2 = []

    idxs_users = [x for x in range(args.num_users)]
    # print(idxs_users)
    # print(dict_users)
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        w_locals.append(copy.deepcopy(w))
    w_locals.append(w_locals[0])

    target_w = []
    targe = dsds_fiveoo()
    # target_w = []

    all_acc = []
    all_loss = []

    for i in range(len(targe)):
        print(time.asctime())
        targe_glob = copy.deepcopy(w_glob)
        net_glob.load_state_dict(copy.deepcopy(targe_glob))
        save_acc = []
        save_loss = []
        loss_train_sds = []
        for ss in range(args.epochs):
            w_locals2 = []

            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(targe[i], m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals2.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            loss_avg = sum(loss_locals) / len(loss_locals)
            print(loss_avg)
            loss_train_sds.append(loss_avg)
            targe_glob = FedAvg(w_locals2)
            net_glob.load_state_dict(copy.deepcopy(targe_glob))

        target_w.append(copy.deepcopy(targe_glob))
        acc_train, loss_train22 = test_img(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("________________________________")


        plt.figure()
        plt.plot(range(len(loss_train_sds)), loss_train_sds)
        plt.ylabel('train_loss')
        plt.savefig('./save/mnist5_2/fedCSA{}.png'.format(i))

    result = FedAvg(target_w)
    net_glob.load_state_dict(result)

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    for i in range(len(target_w)):
        if not i:
            ress = target_w[1:]
        elif (i + 1) == len(target_w):
            ress = target_w[:-1]
        else:
            ress = target_w[:i] + target_w[i + 1:]
        sdsd = FedAvg(ress)
        net_glob.load_state_dict(sdsd)
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("--------------------------------------")
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
    print("over")
