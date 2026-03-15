import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader
from pathlib import Path
import zipfile
import requests
import re
import py7zr


# https://github.com/FengHZ/KD3A/blob/master/datasets/DigitFive.py
def _resolve_data_file(base_path, filename):
    direct_path = path.join(base_path, filename)
    if path.exists(direct_path):
        return direct_path

    matches = list(Path(base_path).rglob(filename))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"Cannot find required file '{filename}' under '{base_path}'")


def load_mnist(base_path):
    print("load mnist")
    mnist_data = loadmat(_resolve_data_file(base_path, "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path):
    print("load mnist_m")
    mnistm_data = loadmat(_resolve_data_file(base_path, "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn(base_path):
    print("load svhn")
    svhn_train_data = loadmat(_resolve_data_file(base_path, "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(_resolve_data_file(base_path, "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]
    svhn_train = svhn_train[:25000]
    train_label = train_label[:25000]
    svhn_test = svhn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_test, test_label


def load_syn(base_path):
    print("load syn")
    syn_train_data = loadmat(_resolve_data_file(base_path, "synth_train_32x32.mat"))
    syn_test_data = loadmat(_resolve_data_file(base_path, "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train[:25000]
    syn_test = syn_test[:9000]
    train_label = train_label[:25000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_test, test_label


def load_usps(base_path):
    print("load usps")
    usps_dataset = loadmat(_resolve_data_file(base_path, "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train = np.tile(usps_train, (4, 1, 1, 1))
    train_label = np.tile(train_label,4)
    usps_train = usps_train[:25000]
    train_label = train_label[:25000]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_test, test_label

class Digit5Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(Digit5Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]

def digit5_dataset_read(base_path, domain):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = Digit5Dataset(data=train_image, labels=train_label, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = Digit5Dataset(data=test_image, labels=test_label, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "Digit5/"
dir_path = "Digit5/"


def _download_from_google_drive(file_id, archive_path):
    base_url = "https://drive.google.com/uc"
    session = requests.Session()
    params = {"export": "download", "id": file_id}

    first = session.get(base_url, params=params, timeout=60)
    first.raise_for_status()

    download_url = first.url
    download_params = None
    content_type = (first.headers.get("Content-Type") or "").lower()

    if "text/html" in content_type:
        html = first.text
        form_match = re.search(
            r'<form id="download-form" action="([^"]+)" method="get">(.*?)</form>',
            html,
            flags=re.S,
        )
        if not form_match:
            return False
        download_url = form_match.group(1)
        form_html = form_match.group(2)
        hidden_inputs = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', form_html))
        if "id" not in hidden_inputs:
            hidden_inputs["id"] = file_id
        hidden_inputs.setdefault("export", "download")
        download_params = hidden_inputs

    response = session.get(download_url, params=download_params, stream=True, timeout=60)
    response.raise_for_status()

    tmp_path = archive_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not (_detect_archive_format(tmp_path) in {"zip", "7z"}):
        os.remove(tmp_path)
        return False

    os.replace(tmp_path, archive_path)
    return True


def _detect_archive_format(archive_path):
    if zipfile.is_zipfile(archive_path):
        return "zip"
    try:
        with open(archive_path, "rb") as f:
            if f.read(6) == b"7z\xbc\xaf\x27\x1c":
                return "7z"
    except OSError:
        return None
    try:
        if py7zr.is_7zfile(archive_path):
            return "7z"
    except Exception:
        return None
    return None


def _download_digit5_archive(archive_path):
    # Original ID is deprecated in some mirrors; the second one is from KD3A README.
    candidate_ids = [
        "1PT6K-_wmsUEUCxoYzDy0mxF-15tvb2Eu",
        "1QvC6mDVN25VArmTuSHqgd7Cf9CoiHvVt",
    ]

    for file_id in candidate_ids:
        try:
            print(f"Downloading Digit5 archive from Google Drive file id: {file_id}")
            if _download_from_google_drive(file_id, archive_path):
                return
            print(f"Downloaded file from id={file_id} is not a valid archive.")
        except Exception as exc:
            print(f"Download failed for id={file_id}: {exc}")

    raise RuntimeError(
        "Failed to download a valid Digit5.zip archive automatically. "
        f"Please place Digit5.zip under {path.dirname(archive_path)} manually."
    )


# Allocate data to usersz``
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"
    
    # Get Digit5 data
    if not os.path.exists(root):
        os.makedirs(root)
    archive_path = path.join(root, "Digit5.zip")
    required_file = path.join(root, "mnistm_with_label.mat")
    if not path.exists(required_file):
        archive_format = _detect_archive_format(archive_path) if path.exists(archive_path) else None
        if archive_format is None:
            _download_digit5_archive(archive_path)
            archive_format = _detect_archive_format(archive_path)
        if archive_format == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(root)
        elif archive_format == "7z":
            with py7zr.SevenZipFile(archive_path, "r") as zf:
                zf.extractall(path=root)
        else:
            raise RuntimeError(f"Unsupported Digit5 archive format: {archive_path}")

    X, y = [], []
    domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
    for d in domains:
        train_loader, test_loader = digit5_dataset_read(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))
    

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
        statistic, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)
