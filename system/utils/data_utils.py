import numpy as np
import os
import torch
from collections import defaultdict
from functools import lru_cache


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


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

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

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
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
