import copy
import glob
import os
import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
from PIL import Image

_ENCODING_IMPORT_ERROR = None
try:
    import encoding.datasets as enc_ds
except Exception as exc:
    enc_ds = None
    _ENCODING_IMPORT_ERROR = exc

if enc_ds is not None:
    encoding_datasets = {
        x: functools.partial(enc_ds.get_dataset, x)
        for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
    }
else:
    encoding_datasets = {}


class FolderLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = get_folder_images(root)
        # Keep ADE20K class count expected by downstream visualization logic.
        self.num_class = 150
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.images[index])

    def make_pred(self, pred):
        """Mimic ADE20K-style prediction ids where class indices start from 1."""
        return np.asarray(pred) + 1

    def __len__(self):
        return len(self.images)


def get_folder_images(img_folder):
    # Windows glob matching is case-insensitive, so we deduplicate by normalized path.
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    glist = []
    folder = img_folder.rstrip("/")
    for pattern in patterns:
        glist.extend(glob.glob(os.path.join(folder, pattern)))

    deduped = {}
    for path in glist:
        deduped[os.path.normcase(os.path.abspath(path))] = path

    return sorted(deduped.values())


def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    if enc_ds is None:
        raise RuntimeError(
            "PyTorch-Encoding is unavailable, only folder-based inference datasets are supported."
        ) from _ENCODING_IMPORT_ERROR
    assert False, f"dataset {name} not found"


def get_original_dataset(name, **kwargs):
    if os.path.isdir(name):
        print("load", name, "as image directroy for FolderLoader")
        return FolderLoader(name, transform=kwargs["transform"])
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys())
