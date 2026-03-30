from __future__ import annotations

import copy
import glob
import random
from typing import Callable
import ast
import webdataset as wds
import wordsegment as ws
from loguru import logger
from torch.utils.data import IterableDataset
from torchvision import transforms as T
from torch.nn import Module
import torch
from torch.nn.utils.rnn import pad_sequence

import uncha.utils.distributed as dist
import torchvision.transforms.functional as F

ws.load()

class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from a lot of TAR file shards.
    This class uses `WebDataset <https://github.com/webdataset/webdataset>`_
    internally, and expects TAR files to be arranged in a compatible format.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal buffer of instances. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        # Convert a single path (glob) to a list.
        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns to list a full list of individual TAR files.
        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        # Sort paths; webdataset performs a deterministic shuffle (internally).
        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        # Shard the TAR file paths as per number of GPU processes to avoid loading
        # duplicates.
        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            # Run for one epoch and stop:
            yield from pipeline

def unnormalize(img, mean, std):
    # img: Tensor [C,H,W]
    mean = torch.tensor(mean).view(3,1,1).to(img.device)
    std = torch.tensor(std).view(3,1,1).to(img.device)
    return img * std + mean

class TransformWithBoxes:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, boxes):

        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            w, h = image.size

        boxes = boxes.clone()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        if isinstance(image, torch.Tensor):
            _, old_h, old_w = image.shape
        else:
            old_w, old_h = image.size

        if isinstance(self.transform, T.Resize):
            if isinstance(self.transform.size, (tuple, list)):
                new_h, new_w = self.transform.size
            else:
                new_h = new_w = self.transform.size
            scale_x, scale_y = new_w / old_w, new_h / old_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            image = self.transform(image)

        elif isinstance(self.transform, T.RandomResizedCrop):
            i, j, h_crop, w_crop = self.transform.get_params(
                image, self.transform.scale, self.transform.ratio
            )
            image = F.resized_crop(image, i, j, h_crop, w_crop,
                                   self.transform.size, self.transform.interpolation)

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=j, max=j + w_crop) - j
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=i, max=i + h_crop) - i

            if isinstance(self.transform.size, (tuple, list)):
                out_h, out_w = self.transform.size
            else:
                out_h = out_w = self.transform.size
            scale_x = out_w / w_crop
            scale_y = out_h / h_crop
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        elif isinstance(self.transform, T.CenterCrop):
            if isinstance(self.transform.size, (tuple, list)):
                th, tw = self.transform.size
            else:
                th = tw = self.transform.size
            i = int(round((old_h - th) / 2.))
            j = int(round((old_w - tw) / 2.))
            image = F.crop(image, i, j, th, tw)

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=j, max=j + tw) - j
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=i, max=i + th) - i

        elif isinstance(self.transform, (T.ColorJitter, T.RandomGrayscale,
                                         T.ToTensor, T.Normalize, T.Lambda)):
            image = self.transform(image)

        if isinstance(image, torch.Tensor):
            _, out_h, out_w = image.shape
        else:
            out_w, out_h = image.size

        boxes[:, [0, 2]] /= max(out_w, 1e-6)
        boxes[:, [1, 3]] /= max(out_h, 1e-6)
        return image, boxes


class GroundedDatasetTarMapper:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        max_boxes: int = 5,        
        tokenizer = None
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
        """
        self.image_transform = T.Compose(image_transform)
        self.max_boxes = max_boxes
        self.tokenizer = tokenizer
        wrapped = []
        for tr in image_transform:
            wrapped.append(TransformWithBoxes(tr))
        self.image_transform = wrapped


    def apply_transforms(self, image, boxes):
        for tr in self.image_transform:
            image, boxes = tr(image, boxes)
        return image, boxes

    def __call__(self, dataset_dict: dict):
        image_orig = dataset_dict["child.jpg"]

        whole_image = image_orig
        text = dataset_dict["child.txt"]
        num_boxes = int(dataset_dict["numparents.txt"])
        
        pos_box_images_list, box_images_lst, box_texts_lst, boxes_lst = [], [], [], []
        for i in range(num_boxes):
            key = f"parent{i:03d}"

            box_images_lst.append(self.apply_transforms(dataset_dict[f"{key}.jpg"], torch.zeros((0,4)))[0])
       
            box_texts_lst.append(dataset_dict[f"{key}.txt"])
            box_str = dataset_dict[f"{key}_box.txt"] 
            box_list = ast.literal_eval(box_str)

            box_tensor = torch.tensor(box_list, dtype=torch.float32)
            boxes_lst.append(box_tensor)

        boxes_stack = torch.stack(boxes_lst, dim=0)
        image, boxes_after = self.apply_transforms(image_orig, boxes_stack)

        limit = min(self.max_boxes, num_boxes)

        box_images = torch.zeros((self.max_boxes, 3, 224, 224))
        box_images[:limit] = torch.stack(box_images_lst[:limit])

        boxes_template = torch.zeros((self.max_boxes, 5))
 

        if boxes_after.sum() < 0.05 or len(boxes_lst) == 0:
            
            boxes_after = torch.tensor([0.25, 0.25, 0.75, 0.75], dtype=torch.float32).unsqueeze(0).repeat(5, 1)
                    

        try: 
            boxes_template[:limit, :4] = boxes_after[:limit]
        except:
            print(F"boxes_template: {boxes_template.shape}. boxes_after: {boxes_after.shape}")

        boxes_template[:limit, 4] = 1
        to_tensor = T.ToTensor()

        return {
            "__key__": dataset_dict["__key__"],
            "image": image,
            "text": text,
            "num_boxes": num_boxes,
            "box_image": box_images,
            "box_text": box_texts_lst,
            "bbox": boxes_template,
        }


