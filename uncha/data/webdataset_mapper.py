#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator, Tuple

import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import Dataset, IterDataPipe
from torch.utils.data.datapipes.iter import ShardingFilter
from torchvision.datasets import ImageFolder

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm
from itertools import islice

class CocoCaptions(Dataset):
    """
    COCO captions dataset. Homepage: https://cocodataset.org
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        """
        Args:
            root: Dataset root directory. It should contain image directories
                named `train2017` and `val2017`, and a separate directory
                containing caption annotations JSON file.
            split: Name of 2017 split to load, one of `{train, val}`.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Read annotations for the given split.
        json_path = self.root / "annotations" / f"captions_{split}2017.json"
        coco_json = json.load(open(json_path))

        # Build a temporary mapping between image ID and captions.
        image_id_to_anns = defaultdict(list)
        for ann in coco_json["annotations"]:
            image_id_to_anns[ann["image_id"]].append(ann)

        # Convert the above mapping to list of tuples formatted as:
        # `(image_id, image_path, list[caption_ids], list[caption])`.
        self.samples = [
            (
                image_id,
                self.root / f"{split}2017" / f"{image_id:0>12d}.jpg",
                [ann["id"] for ann in anns],
                [ann["caption"] for ann in anns],
            )
            for image_id, anns in image_id_to_anns.items()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_id, image_path, caption_ids, captions = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image_id": image_id,
            "caption_ids": caption_ids,
            "image": image,
            "captions": captions,
        }


class Flickr30kCaptions(CocoCaptions):
    """
    Flickr30K captions dataset.

    Karpathy split JSON can be downloaded from this webpage:
    https://cs.stanford.edu/people/karpathy/deepimagesent/
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        """
        Args:
            root: Dataset root directory. It should contain a JSON file named
                `dataset_flickr30k.json` containing Karpathy splits, and a
                directory named `flickr30k_images` with all images (~31K).
            split: Name of split to load, one of `{train, val, test}`.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Read annotations and keep only those belonging to specified split.
        flickr_json = json.load(open(self.root / "dataset_flickr30k.json"))

        # Convert the filtered list of tuples formatted as:
        # `(image_id, image_path, list[caption_ids], list[caption])`.
        # Only keep images that belong to required split.
        self.samples = [
            (
                int(ann["filename"][:-4]),
                self.root / "flickr30k_images" / ann["filename"],
                ann["sentids"],
                [entry["raw"] for entry in ann["sentences"]],
            )
            for ann in flickr_json["images"]
            if ann["split"] == split
        ]


class ImageNet(ImageFolder):
    """
    Lightweight wrapper over Torchvision `ImageFolder` to load ImageNet dataset.
    """

    def __init__(self, root: str, split: str = "train", **kwargs):
        # super().__init__(str(Path(root) / split), **kwargs)
        super().__init__(os.path.join("/path/to/your/path", split), **kwargs)
        


class CUB2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, split, transform=None, target_transform=None, download=True):
        super(CUB2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.split = split
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.split == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class TfdsWrapper(IterDataPipe):
    """
    Minimal wrapper on `tensorflow-datasets` to serve `(image, label)`
    tuples for image classification datasets. This wrapper enables a consistent
    output format with dataset implementations from the Torchvision library.
    """

    def __init__(
        self,
        name: str,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
    ):
        """
        Args:
            name: Name of a dataset supported by Tensorflow datasets. See
                https://www.tensorflow.org/datasets/catalog/overview for details.
            root: Dataset root directory. This is passed to the `data_dir`
                argument of `tfds.load`. All datasets are auto-downloaded and
                cached in this directory.
            split: Which dataset split to load. This should be one of the official
                splits for the given dataset.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """

        super().__init__()
        self.name = name
        self.split = split
        self.transform = transform
        
        if name == "caltech101":
            name = "caltech101:3.0.2"

        dset = tfds.load(name, split=split, data_dir=root, download=False)

        dset = tfds.as_numpy(dset)

        # Record length of the dataset before further wrapping.
        self._length = len(dset)

        # Wrap the tensorflow dataset with `IterDataPipe` and apply sharding filter
        # to avoid duplicates when multiple CPU workers are used in DataLoader.
        self.dset = ShardingFilter(dset)

    def __repr__(self):
        return f"TfDatasetWrapper(name={self.name}, split={self.split})"

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            # Convert numpy arrays: image (PIL.Image) and label (tensor).
            # Handle special case with MNIST images.
            if self.name == "mnist":
                image = Image.fromarray(instance["image"][..., 0], mode="L")
            else:
                image = Image.fromarray(instance["image"])

            image = image.convert("RGB")
            label = torch.tensor(instance["label"])

            if self.transform is not None:
                image = self.transform(image)

            yield image, label


class CLEVRCounts(TfdsWrapper):
    """
    CLEVR-Counts image classification dataset. Counting the number of objects in
    a scene is framed as a classification task. This task was included in the
    Visual Task Adaptation Benchmark (VTAB), and used in CLIP evaluation suite.
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        super().__init__("clevr", root, split, transform)

        # Convert counts to contiguous labels.
        self._labels = [10, 3, 4, 5, 6, 7, 8, 9]

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            image = Image.fromarray(instance["image"]).convert("RGB")
            num_objects = len(instance["objects"]["color"])
            label = torch.tensor(self._labels.index(num_objects))

            if self.transform is not None:
                image = self.transform(image)

            yield image, label