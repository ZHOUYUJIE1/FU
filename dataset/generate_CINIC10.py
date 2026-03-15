import numpy as np
import os
import random
import shlex
import sys
import torch
import torchvision.transforms as transforms

from io import BytesIO
from pathlib import Path
from PIL import Image
from torchvision.datasets import ImageFolder
from utils.dataset_utils import check, save_file, separate_data, split_data


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "CINIC10/"
CINIC10_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
HF_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/flwrlabs/cinic10/resolve/main/data/train-00000-of-00001.parquet",
    "valid": "https://huggingface.co/datasets/flwrlabs/cinic10/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/flwrlabs/cinic10/resolve/main/data/test-00000-of-00001.parquet",
}


def _resolve_cinic_root(rawdata_dir):
    candidates = [
        rawdata_dir / "CINIC-10",
        rawdata_dir / "cinic-10",
        rawdata_dir / "CINIC10",
        rawdata_dir / "cinic10",
        rawdata_dir,
    ]

    for root in candidates:
        if (root / "train").exists() and (root / "test").exists():
            return root
    raise FileNotFoundError(
        "CINIC-10 raw data not found. Expected a directory containing train/test "
        f"under {rawdata_dir} (for example: {rawdata_dir / 'CINIC-10'})."
    )


def _download_and_extract_cinic(rawdata_dir):
    rawdata_dir.mkdir(parents=True, exist_ok=True)
    archive_path = rawdata_dir / "CINIC-10.tar.gz"

    if not (rawdata_dir / "CINIC-10").exists():
        download_cmd = (
            "env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY "
            f"wget -c -O {shlex.quote(str(archive_path))} {shlex.quote(CINIC10_URL)}"
        )
        if os.system(download_cmd) != 0:
            raise RuntimeError(f"Failed to download CINIC-10 archive from {CINIC10_URL}")

    if not (rawdata_dir / "CINIC-10").exists():
        extract_cmd = f"tar -xzf {shlex.quote(str(archive_path))} -C {shlex.quote(str(rawdata_dir))}"
        if os.system(extract_cmd) != 0:
            raise RuntimeError(f"Failed to extract CINIC-10 archive: {archive_path}")


def _resolve_hf_parquet_root(rawdata_dir):
    candidates = [
        rawdata_dir / "hf_parquet",
        rawdata_dir,
    ]
    for root in candidates:
        if all((root / f"{split}.parquet").exists() for split in ("train", "valid", "test")):
            return root
    raise FileNotFoundError("Hugging Face parquet files not found.")


def _download_hf_parquet(rawdata_dir):
    parquet_root = rawdata_dir / "hf_parquet"
    parquet_root.mkdir(parents=True, exist_ok=True)
    for split, url in HF_PARQUET_URLS.items():
        output_path = parquet_root / f"{split}.parquet"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        download_cmd = (
            "env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY "
            f"curl -L --fail --retry 3 -C - -o {shlex.quote(str(output_path))} {shlex.quote(url)}"
        )
        if os.system(download_cmd) != 0:
            raise RuntimeError(f"Failed to download CINIC-10 parquet split '{split}' from {url}")
    return parquet_root


def _load_hf_parquet_dataset(parquet_root, transform):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading CINIC-10 parquet splits requires pyarrow.") from exc

    dataset_image = []
    dataset_label = []

    for split in ("train", "valid", "test"):
        parquet_path = parquet_root / f"{split}.parquet"
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=256):
            batch_dict = batch.to_pydict()
            image_column = batch_dict.get("image") or batch_dict.get("img")
            label_column = batch_dict.get("label") or batch_dict.get("fine_label")
            if image_column is None or label_column is None:
                raise RuntimeError(f"Unsupported CINIC-10 parquet schema in {parquet_path}.")

            for image_item, label_item in zip(image_column, label_column):
                if isinstance(image_item, dict):
                    image_bytes = image_item.get("bytes")
                    image_path = image_item.get("path")
                    if image_bytes is not None:
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    elif image_path:
                        image = Image.open(image_path).convert("RGB")
                    else:
                        raise RuntimeError(f"Unsupported image payload in {parquet_path}.")
                elif isinstance(image_item, (bytes, bytearray)):
                    image = Image.open(BytesIO(image_item)).convert("RGB")
                else:
                    image = Image.fromarray(np.asarray(image_item, dtype=np.uint8)).convert("RGB")

                dataset_image.append(transform(image).cpu().numpy())
                dataset_label.append(int(label_item))

    if not dataset_image:
        raise RuntimeError(f"No CINIC-10 samples loaded from parquet root {parquet_root}.")

    return np.stack(dataset_image, axis=0), np.asarray(dataset_label)


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

    rawdata_dir = Path(dir_path) / "rawdata"

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_image = None
    dataset_label = None
    cinic_root = None

    try:
        cinic_root = _resolve_cinic_root(rawdata_dir)
    except FileNotFoundError:
        cinic_root = None

    if cinic_root is not None:
        dataset_image = []
        dataset_label = []
        for split in ("train", "valid", "test"):
            split_root = cinic_root / split
            if not split_root.exists():
                continue
            split_set = ImageFolder(root=str(split_root), transform=transform)
            split_loader = torch.utils.data.DataLoader(split_set, batch_size=512, shuffle=False)
            for x, y in split_loader:
                dataset_image.append(x.cpu().numpy())
                dataset_label.append(y.cpu().numpy())

        if dataset_image:
            dataset_image = np.concatenate(dataset_image, axis=0)
            dataset_label = np.concatenate(dataset_label, axis=0)

    if dataset_image is None or dataset_label is None:
        try:
            parquet_root = _resolve_hf_parquet_root(rawdata_dir)
        except FileNotFoundError:
            print("CINIC-10 raw data not found locally. Downloading Hugging Face parquet fallback...")
            parquet_root = _download_hf_parquet(rawdata_dir)
        dataset_image, dataset_label = _load_hf_parquet_dataset(parquet_root, transform)

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
