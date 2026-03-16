import numpy as np
import os
import torch
from collections import defaultdict
from functools import lru_cache
from torchvision import transforms


_CIFAR_SPATIAL_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])
_CIFAR_MEAN = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)
_CIFAR_STD = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(3, 1, 1)


@lru_cache(maxsize=256)
def _read_npz_payload(file_path):
    """Cache decompressed client payloads to avoid repeated zip/pickle reads."""
    with open(file_path, 'rb') as f:
        return np.load(f, allow_pickle=True)['data'].tolist()


def read_data(dataset, idx, is_train=True):
    # 以当前文件为基准，定位到项目根目录，避免工作目录不同导致的相对路径问题
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    split = 'train' if is_train else 'test'
    data_dir = os.path.join(project_root, 'dataset', dataset, split)

    file = os.path.join(data_dir, f"{idx}.npz")
    return _read_npz_payload(file)


def clear_data_cache():
    _read_npz_payload.cache_clear()


def _apply_cifar_train_transform(x):
    # CIFAR tensors are stored after Normalize(0.5, 0.5, 0.5); restore to [0, 1]
    # before spatial augmentation, then normalize back to the training scale.
    mean = _CIFAR_MEAN.to(device=x.device, dtype=x.dtype)
    std = _CIFAR_STD.to(device=x.device, dtype=x.dtype)
    x = x * std + mean
    x = x.clamp_(0.0, 1.0)
    x = _CIFAR_SPATIAL_TRAIN_TRANSFORM(x)
    x = (x - mean) / std
    return x


def read_client_data(dataset, idx, is_train=True, few_shot=0, apply_train_transform=True):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(
            data,
            dataset=dataset,
            is_train=is_train,
            apply_train_transform=apply_train_transform,
        )

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data, dataset="", is_train=True, apply_train_transform=True):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    if apply_train_transform and is_train and "Cifar" in dataset:
        return [(_apply_cifar_train_transform(x), label) for x, label in zip(X, y)]
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]
