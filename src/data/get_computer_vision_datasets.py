import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10, MNIST, SVHN, CelebA, FashionMNIST
import subprocess
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

class CelebANoIntegrityCheck(CelebA):
    def _check_integrity(self) -> bool:
        return True

def download_celeba_alternative(data_dir):
    """Attempts to download CelebA from alternative sources if it's not found."""
    img_path = data_dir / "celeba" / "img_align_celeba"

    if img_path.exists():
        print("CelebA dataset found locally. Skipping download.")
        return

    print("Dataset not found. Attempting alternative sources...")

    try:
        zip_file_path = data_dir / "celeba" / "img_align_celeba.zip"

        if not zip_file_path.exists():
            print("Downloading CelebA dataset...")
            url = "https://cseweb.ucsd.edu/~weijian/static/datasets/celeba/img_align_celeba.zip"
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(zip_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")
        else:
            print("ZIP file already exists. Skipping download.")

        print("Extracting CelebA dataset...")
        with ZipFile(zip_file_path, 'r') as zfile:
            zfile.extractall(img_path)
        print("Dataset extracted successfully.")

        # Move files from nested folder to desired folder structure
        nested_dir = img_path / "img_align_celeba"

        if nested_dir.exists():
            for item in nested_dir.iterdir():
                shutil.move(str(item), img_path)
            nested_dir.rmdir()  # Remove empty nested directory

        print("CelebA downloaded successfully.")
        return
    except Exception as e:
        print("CelebA download failed:", str(e))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )
    parser.add_argument(
        "--download_celeba",
        type=bool,
        default=True,
        help="Will attempt to download the CelebA dataset." " Set to False if manually downloaded.",
    )
    args = parser.parse_args()
    return args


def download_data(data_root, download_celeba):
    # MNIST
    MNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = MNIST(root=data_root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # FashionMNIST
    FashionMNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = FashionMNIST(root=data_root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # CIFAR10
    root = Path(data_root) / "CIFAR10" / "raw"
    CIFAR10(root, download=True)
    for set in ["train", "test"]:
        dataset = CIFAR10(root=root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # SHVN
    root = Path(data_root) / "SVHN" / "raw"
    for set in ["train", "test"]:
        dataset = SVHN(root=root, split=set, download=True)
        dataset_name = dataset.__class__.__name__
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # CelebA
    root = Path(data_root) / "CelebA" / "raw"

    download_celeba_alternative(root)

    for set in ["train", "valid", "test"]:
        dataset = CelebANoIntegrityCheck(root=root, split=set, download=False)
        # dataset = CelebA(root=root, split=set, download=False, check_integrity=False)
        dataset_name = dataset.__class__.__name__
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.resize((32, 32))
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


def create_train_test_splits(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    # need to create a train/val split for these datasets
    for dataset in ["FashionMNIST", "MNIST", "CIFAR10", "SVHN"]:
        numpy_data_root = Path(data_root) / dataset / "numpy"
        train_and_val_list = list((numpy_data_root / "train").glob("*"))
        train_list, val_list = train_test_split(train_and_val_list, test_size=0.05, random_state=42)
        test_list = list((numpy_data_root / "test").glob("*"))
        for split_name, data_split in zip(
            ["train", "val", "test"], [train_list, val_list, test_list]
        ):
            save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")

    # CelebA already has a train/val split
    dataset = "CelebA"
    numpy_data_root = Path(data_root) / dataset / "numpy"
    train_list = list((numpy_data_root / "train").glob("*"))
    val_list = list((numpy_data_root / "valid").glob("*"))
    test_list = list((numpy_data_root / "test").glob("*"))
    for split_name, data_split in zip(["train", "val", "test"], [train_list, val_list, test_list]):
        save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")


def main():
    args = parse_args()
    download_data(data_root=args.data_root, download_celeba=args.download_celeba)
    create_train_test_splits(data_root=args.data_root)

if __name__ == "__main__":
    main()
