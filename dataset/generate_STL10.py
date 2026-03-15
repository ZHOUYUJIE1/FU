import numpy as np
import os
import random
import sys
import torch
import torchvision
import torchvision.transforms as transforms

from utils.dataset_utils import check, save_file, separate_data, split_data


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "STL10/"


def generate_dataset(dir_path, num_clients, niid, balance, partition, class_per_client=2):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    effective_class_per_client = class_per_client if partition in ("pat", "exdir") else None
    if check(
        config_path,
        train_path,
        test_path,
        num_clients,
        niid,
        balance,
        partition,
        class_per_client=effective_class_per_client,
    ):
        return

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_image = []
    dataset_label = []

    for split in ("train", "test"):
        split_set = torchvision.datasets.STL10(
            root=dir_path + "rawdata",
            split=split,
            download=True,
            transform=transform,
        )
        split_loader = torch.utils.data.DataLoader(split_set, batch_size=1024, shuffle=False)
        for x, y in split_loader:
            dataset_image.append(x.cpu().numpy())
            dataset_label.append(y.cpu().numpy())

    dataset_image = np.concatenate(dataset_image, axis=0)
    dataset_label = np.concatenate(dataset_label, axis=0)

    num_classes = len(set(dataset_label.tolist()))
    print(f"Number of classes: {num_classes}")

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=class_per_client,
    )
    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
        class_per_client=effective_class_per_client,
    )


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
