import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST, CIFAR100, ImageFolder, DatasetFolder, utils
import torch.utils.data as data
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np


def non_iid_two():
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())

    # 定义不同的数据分布
    num_classes = 10
    class_distribution = [0.1, 0.2, 0.3, 0.4]

    # 将数据集分成多个设备
    num_devices = len(class_distribution)
    device_datasets = []
    for i in range(num_devices):
        # 在数据集中选择特定的类别
        classes = torch.tensor([j for j in range(num_classes) if j % num_devices == i])
        indices = torch.tensor([k for k in range(len(train_dataset)) if train_dataset[k][1] in classes])
        device_dataset = torch.utils.data.Subset(train_dataset, indices)

        # 对数据集进行重采样，使每个设备的数据分布不同
        weights = torch.tensor([class_distribution[i] if train_dataset[k][1] in classes else 0.0 for k in indices])
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        device_dataset = torch.utils.data.DataLoader(device_dataset, batch_size=32, sampler=sampler)

        device_datasets.append(device_dataset)



class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # if self.train:
        #     data = mnist_dataobj.train_data
        #     target = mnist_dataobj.train_labels
        # else:
        #     data = mnist_dataobj.test_data
        #     target = mnist_dataobj.test_labels

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def non_iid(n_parties, beta, y_train):
    transform = transforms.Compose([transforms.ToTensor()])

    # mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    min_size = 0
    min_require_size = 10
    K = 10

    N = y_train.shape[0]
    # np.random.seed(2020)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]


