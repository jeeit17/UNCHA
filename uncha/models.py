
#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import uncha.utils.distributed as dist
from uncha import lorentz as L
from uncha.encoders.text_encoders import TransformerTextEncoder
import numpy as np
from uncha.tokenizer import Tokenizer
import ast
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CLIPBaseline(nn.Module):
    """
    Re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.global_logit_scale = nn.Parameter(torch.tensor(1/0.07).log())
        self.global_local_logit_scale = nn.Parameter(torch.tensor(1/0.06).log()) 
        self.local_logit_scale = nn.Parameter(torch.tensor(1/0.05).log()) 

        self.tokenizer = Tokenizer()


        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

   
    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of 1D torch.Tensors, each containing token IDs of varying length.
            project: Whether to apply L2 normalization to output features.
        """
        context_len = self.textual.context_length  # e.g., 77
        batch_size = len(tokens)

        padded_tokens = torch.zeros((batch_size, context_len), dtype=torch.long)

        for idx, inst_tokens in enumerate(tokens):
            L = min(inst_tokens.shape[0], context_len)
            if inst_tokens.shape[0] > context_len:
                inst_tokens = inst_tokens[:context_len]
                inst_tokens[-1] = inst_tokens[-1]  # EOS token
            padded_tokens[idx, :L] = inst_tokens[:L]

        padded_tokens = padded_tokens.to(self.device)

        text_feats = self.textual(padded_tokens)

        eos_indices = padded_tokens.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=self.device)
        text_feats = text_feats[batch_indices, eos_indices]

        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats
    
    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        loss = 0.5 * (
            F.cross_entropy(image_logits, targets)
            + F.cross_entropy(text_logits, targets)
        )
        output_dict = {
            "loss": loss,
            "logging": {"contrastive_loss": loss, "logit_scale": _scale},
        }
        return output_dict


class MERU(CLIPBaseline):
    """
    Implementation of MERU model that embeds images and text in a hyperbolic space.

    Reference: MERU paper (https://arxiv.org/abs/2304.09172)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())



    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.curv.exp())

        return image_feats
    
    

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)
        # print(f"text_feats:{text_feats[0]}")
        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            # print(f"text_feats_textual_alpha:{text_feats[0]}")
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())
            # print(f"text_feats_exp_map:{text_feats[0]}")
        return text_feats
    
    def encode_local_image(self, image):
        global_feat, feat_flat = self.visual.forward_intermediates(image, norm=True, intermediates_only=True)

        # global: [B, C], full_feat: [B, C, H, W]

        B, C = global_feat.shape

        feat_flat = self.visual_proj(feat_flat)
        global_feat = self.visual_proj(global_feat)
        feat_flat = feat_flat * self.visual_alpha.exp()
        global_feat = global_feat * self.visual_alpha.exp()
        feat_flat = L.exp_map0(feat_flat, self.curv.exp())
        global_feat = L.exp_map0(global_feat, self.curv.exp())
        
        
        return global_feat, feat_flat


    def forward(
        self, images: torch.Tensor,
        tokens: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }



class HyCoCLIP(MERU):

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `MERU`.

        Args:
            use_boxes: Whether to use box images and texts for training.
        """
        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, pixel_mean, pixel_std)
        assert use_boxes, "HyCoCLIP requires box images and texts to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)


        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # inter-modal
            _local_aperture_thresh = 1.2    # intra-modal

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()
            
            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }



class UNCHA(MERU):
    """
    Our UNCHA model, that modifies MERU and CLIP to embed images, texts and their localized box 
    information hierarchically in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, pixel_mean, pixel_std)


    def piecewise_continuous_entail_loss(self, angle, aperture, tau, factor=0.1):
        val = angle - tau * aperture
        return torch.where(val > 0, val + factor * angle, factor * angle)

    def uncertainty_calibrated_entailment_loss(
        self,
        entail_residual,    
        log_uncertainty, 
        alpha=10.0,
        stop_grad=True
    ):
       
        mean_loss = 0.5 * entail_residual

        uncertainty = torch.exp(log_uncertainty)
        uncertainty = torch.clamp(uncertainty, min=1e-6, max=1e6)

        if stop_grad:
            scaled_entail = entail_residual.detach() / (uncertainty + 1e-6)
        else:
            scaled_entail = entail_residual / (uncertainty + 1e-6)

        calibration_term = 0.5 * scaled_entail + 0.5 * log_uncertainty

        prob = torch.softmax(log_uncertainty.flatten(), dim=0)
        entropy = -(prob * torch.log(prob + 1e-8)).sum()

        calibration_loss = alpha * (calibration_term + entropy)

        return mean_loss.mean(), calibration_loss.mean()

    def calculate_uncertainty(self, x):
        r = F.softplus(-torch.norm(x, dim=-1))
        return r

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor], 
        box_infos: list[torch.Tensor], num_boxes: list[torch.Tensor], 
        iteration: int,
        tokenizer

    ) -> dict[str, torch.Tensor]:
        """
        UNCHA (UNcertainty-guided Compositional Hyperbolic Alignment) forward pass.
 
        Args:
            images:      Whole scene image tensor (B, C, H, W)
            box_images:  Part (cropped region) image tensor (B, N, C, H, W) — N is the max number of parts
            tokens:      Text tokens corresponding to the whole scene
            box_tokens:  Text tokens corresponding to each part (parent captions)
            box_infos:   Part bounding box information (position, validity, etc.)
            num_boxes:   Number of valid parts per sample
            iteration:   Current training iteration
            tokenizer:   Text tokenizer
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # build valid part (box) mask
        valid_mask = (box_infos[:, :, 4] == 1.0) # boolean


        with torch.no_grad():
            tokens = tokenizer(tokens)
            box_tokens_list = []

            for bt_list in box_tokens:  # List[str] per sample
                bt_tokens = tokenizer(bt_list)  # dynamic number of parents
                box_tokens_list.append(bt_tokens)  # move to device
            box_tokens = box_tokens_list

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # encode whole scene 
        text_feats = self.encode_text(tokens, project=True)
        image_feats = self.encode_image(images.to(self.device), project=True) # global, local


        B, N, iC, iH, iW = box_images.shape
        C = 512

        # Initialize tensors for part (box) embeddings
        # Allocate (B*N, C) tensors and fill only at valid part positions.
        whole_text_feats = torch.zeros((B * N, C), device=self.device)
        whole_image_feats =  torch.zeros((B * N, C), device=self.device)


        valid_mask = valid_mask.bool().to(self.device)

        box_text_feats = torch.zeros((B * N, C), device=self.device)
        box_img_feats = torch.zeros((B * N, C), device=self.device)

        flat_box_tokens = [box_tokens[b][i] for b in range(B) for i in range(N) if valid_mask[b, i]]
        flat_box_images = box_images[valid_mask].to(self.device).contiguous()  # (M, C, H, W)
       
        # Encode part text/image into hyperbolic space
        flat_text_feats = self.encode_text(flat_box_tokens, project=True)  # (M, D)
        flat_img_feats = self.encode_image(flat_box_images, project=True)  # (M, D)


        # Scatter valid part embeddings back to the correct positions

        valid_positions = valid_mask.nonzero(as_tuple=False)  # (M, 2)
        valid_flat_idx = valid_mask.view(-1).nonzero(as_tuple=False).squeeze(1).to(self.device)
        box_text_feats = box_text_feats.index_copy(0, valid_flat_idx, flat_text_feats)        
        box_img_feats = box_img_feats.index_copy(0, valid_flat_idx, flat_img_feats)

        valid_mask = valid_mask.view(B*N)
    
        # All-gather embeddings across GPUs
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)
        all_bimg_feats = dist.gather_across_processes(box_img_feats)
        all_btext_feats = dist.gather_across_processes(box_text_feats)
        all_valid_mask = dist.gather_across_processes(valid_mask)
        

        all_valid_mask = torch.cat(all_valid_mask, dim=0) # [total,  C] - boolean

        # Build whole ↔ part mapping for global-local contrastive loss
        selected_image_feats = image_feats.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1)
        selected_image_feats = selected_image_feats[valid_mask]

        selected_text_feats = text_feats.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1)
        selected_text_feats = selected_text_feats[valid_mask]

        all_bimg_feats = torch.cat(all_bimg_feats, dim=0)
        ALL_B = all_valid_mask.shape[0] // N  
        all_bimg_feats = all_bimg_feats[all_valid_mask]
        all_btext_feats = torch.cat(all_btext_feats, dim=0)[all_valid_mask]
        
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        selected_all_image_feats = all_image_feats.unsqueeze(1).expand(ALL_B, N, -1).reshape(ALL_B*N, -1)
        selected_all_image_feats = selected_all_image_feats[all_valid_mask]
        selected_all_text_feats = all_text_feats.unsqueeze(1).expand(ALL_B, N, -1).reshape(ALL_B*N, -1)
        selected_all_text_feats = selected_all_text_feats[all_valid_mask]
        
        # Extract valid part embeddings for the current process
        box_text_feats = box_text_feats[valid_mask]
        box_img_feats = box_img_feats[valid_mask]


        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):

            # Global contrastive loss
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

            # Uncertainty estimation
            # Larger hyperbolic radius (farther from origin) → lower uncertainty
            # Smaller hyperbolic radius (closer to origin) → higher uncertainty
            uncert_box_pair = self.calculate_uncertainty(box_img_feats).detach()  # (M,)
            uncert_text_pair = self.calculate_uncertainty(box_text_feats).detach()  # (M,)

            # Uncertainty-guided temperature scaling
            uncert_temp_scale_box = torch.exp(-0.5 * uncert_box_pair).clamp(min=0.1, max=10.0)  # (M,)
            uncert_temp_scale_txt = torch.exp(-0.5 * uncert_text_pair).clamp(min=0.1, max=10.0)  # (M,)

            # Global-local contrastive logits 
            base_image_logits = -L.pairwise_dist(box_img_feats, selected_all_text_feats, _curv)
            base_text_logits = -L.pairwise_dist(box_text_feats, selected_all_image_feats, _curv)

            global_local_image_logits = uncert_temp_scale_box.unsqueeze(1) * base_image_logits
            global_local_text_logits = uncert_temp_scale_txt.unsqueeze(1) * base_text_logits
 
            # Local contrastive logits
            box_image_logits = -L.pairwise_dist(box_img_feats, all_btext_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_bimg_feats, _curv)
            shirinked_size = selected_image_feats.shape[0]

            # Target indices for contrastive loss
            targets = torch.arange(shirinked_size, device=image_logits.device)
            offset = all_valid_mask[:B * N * self._rank].sum() 
            targets = targets + offset

            # Targets for the global loss (whole scene level, adjusted by process rank)
            original_targets = torch.arange(B, device=image_logits.device)
            original_targets = original_targets + B * self._rank


            self.global_logit_scale.data = torch.clamp(self.global_logit_scale.data, max=4.6052)
            self.local_logit_scale.data = torch.clamp(self.local_logit_scale.data, max=4.6052)
            self.global_local_logit_scale.data = torch.clamp(self.global_local_logit_scale.data, max=4.6052)

            global_scale = self.global_logit_scale.exp()
            local_scale = self.local_logit_scale.exp()
            global_local_scale = self.global_local_logit_scale.exp()

            # Contrastive loss computation
            global_contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(global_scale * image_logits, original_targets) + 
                nn.functional.cross_entropy(global_scale * text_logits, original_targets)
            )
            
            local_contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(local_scale * box_image_logits, targets) + 
                nn.functional.cross_entropy(local_scale * box_text_logits, targets) 
            )
          
            global_local_contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(global_local_scale * global_local_image_logits, targets) + 
                nn.functional.cross_entropy(global_local_scale * global_local_text_logits, targets)
            )

            # Total contrastive loss 
            contrastive_loss = global_contrastive_loss + local_contrastive_loss + global_local_contrastive_loss


            # Entailment loss computation 

            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_img_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_img_feats, selected_image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_img_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, selected_text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # inter-modal
            _local_aperture_thresh = 1.2    # intra-modal
            
            # Piecewise-continuous entailment loss 

            # Inter-modal: whole text → whole image
            text_image_entail_residual = self.piecewise_continuous_entail_loss(
                _angle, _aperture, _global_aperture_thresh
            )

            # Inter-modal: part text → part image
            box_text_image_entail_residual = self.piecewise_continuous_entail_loss(
                _box_angle, _box_aperture, _global_aperture_thresh
            )

            # Intra-modal image: part image → whole image (+ uncertainty computation)
            cross_image_entail_residual = self.piecewise_continuous_entail_loss(
                _cross_image_angle, _box_image_aperture, _local_aperture_thresh
            )

            log_u_img = self.calculate_uncertainty(box_img_feats)

            # Intra-modal text: part text → whole text (+ uncertainty computation)
            cross_text_entail_residual = self.piecewise_continuous_entail_loss(
                _cross_text_angle, _box_text_aperture, _local_aperture_thresh
            )

            log_u_txt = self.calculate_uncertainty(box_text_feats)

            # Inter-modal entailment
            text_image_entailment_loss = 0.5 * text_image_entail_residual.mean()
            box_text_image_entailment_loss = 0.5 * box_text_image_entail_residual.mean()

            # Intra-modal entailment + Uncertainty calibration

            cross_image_entailment_loss, cross_img_calibration_loss = self. uncertainty_calibrated_entailment_loss(
                    cross_image_entail_residual,
                    log_u_img
                )

            cross_text_entailment_loss, cross_txt_calibration_loss = self.uncertainty_calibrated_entailment_loss(
                cross_text_entail_residual,
                log_u_txt
            )

            entailment_loss = (
                text_image_entailment_loss
                + box_text_image_entailment_loss
                + 0.5 * (cross_image_entailment_loss + cross_text_entailment_loss)
                + cross_img_calibration_loss
                + cross_txt_calibration_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = (
                    loss + 
                    self.entail_weight * entailment_loss 
                )
          
        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "global_scale": global_scale,
                "local_scale": local_scale,
                "global_local_scale": global_local_scale,
                "curv": _curv
            },
        }