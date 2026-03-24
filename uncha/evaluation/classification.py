#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from uncha import lorentz as L
from uncha.evaluation.catalog import DatasetCatalog
from uncha.evaluation.class_names import CLASS_NAMES
from uncha.models import UNCHA, HyCoCLIP, MERU, CLIPBaseline
from uncha.tokenizer import Tokenizer


class ZeroShotClassificationEvaluator:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        self._datasets_and_prompts = datasets_and_prompts
        self._data_dir = Path(data_dir).resolve()
        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(self, model: UNCHA | HyCoCLIP | MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()

        # Collect results per task in this dict:
        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            # ----------------------------------------------------------------
            # Make zero-shot classifier using class name and prompts.
            # ----------------------------------------------------------------
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for i, name in enumerate(class_names):
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)

                if isinstance(model, (UNCHA, HyCoCLIP, MERU)):
                    # Ensemble in the tangent space, then project to Hyperboloid.
                    class_feats = class_feats.mean(dim=0)
                    class_feats = class_feats * model.textual_alpha.exp()
                    class_feats = L.exp_map0(class_feats, model.curv.exp())
                else:
                    # Ensemble prompt features: normalize -> average -> normalize.
                    class_feats = F.normalize(class_feats, dim=-1)
                    class_feats = class_feats.mean(dim=0)
                    class_feats = F.normalize(class_feats, dim=-1)

                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim)
            classifier = torch.stack(all_class_feats, dim=0)
            logger.info(f"Classifier shape: {classifier.size()}")

            # Extract image features and labels from the test split of required dataset.
            loader = DataLoader(
                DatasetCatalog.build(
                    dname, self._data_dir, "test", self._image_transform
                ),
                batch_size=64,
            )
            image_feats, labels = _encode_dataset(loader, model, project=True)

            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)

            # Measure model performance according to accuracy metric of the dataset.
            acc_meter = MulticlassAccuracy(DatasetCatalog.NUM_CLASSES[dname])

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in zip(image_feats.split(256), labels.split(256)):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, (UNCHA, HyCoCLIP, MERU)):
                    scores = L.pairwise_inner(_feats, classifier, model.curv.exp())
                   
                else:
                    scores = _feats @ classifier.T

                acc_meter(scores.cpu(), _labels)

            accuracy = acc_meter.compute() * 100.0
            results_dict[dname] = accuracy

            logger.info(
                f"Zero-shot evaluation: {dname}, {len(image_feats)} images, "
                f"{len(class_names)} classes [acc.: {accuracy:.1f}%] "
            )

        return results_dict


def _encode_dataset(
    data_loader: DataLoader,
    model: UNCHA | HyCoCLIP | MERU | CLIPBaseline,
    project: bool,
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_labels = [], []

    for images, labels in tqdm(data_loader, desc=f"Extracting image feats", ncols=60):
        with torch.inference_mode():
            image_feats = model.encode_image(images.to(model.device), project)

        # all_image_feats.append(image_feats.cpu())
        all_image_feats.append(image_feats)
        all_labels.append(labels)

    logger.info(f"Extracted {len(all_image_feats)} batches of image features.")
    return torch.cat(all_image_feats, dim=0), torch.cat(all_labels, dim=0)
