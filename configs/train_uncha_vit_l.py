#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Each config file should have four dicts or OmegaConf objects:
`dataset`, `model`, `optim`, and `train`.

User can compose config files by importing these objects and overriding specific
parameters. See examples in other training configs.

Reference: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
"""

from torch.optim import AdamW
from torchvision import transforms as T

from uncha.config import LazyCall as L
from uncha.data.webdataset_mapper import GroundedDatasetTarMapper, ImageTextWebDataset
from uncha.encoders.image_encoders import build_timm_vit
from uncha.encoders.text_encoders import TransformerTextEncoder
from uncha.models import UNCHA
from uncha.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


dataset = L(ImageTextWebDataset)(
    tarfiles=["/path/to/your/tarfiles"],


    mapper=L(GroundedDatasetTarMapper)(
        image_transform = [
            L(T.RandomResizedCrop)(
                size=224,
                scale=(0.8, 1.0),        
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            L(T.RandomApply)(
                transforms=[
                    T.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            L(T.RandomGrayscale)(p=0.2),
            L(T.ToTensor)(),
        ]
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)


model = L(UNCHA)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
        
    ),

    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77 # originally context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
    use_boxes=True,
)


optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=[
                "global_logit_scale", "local_logit_scale", "global_local_logit_scale", 
                "visual_alpha", "textual_alpha", "curv"
            ],
        ),
        lr=5e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


train = dict(
    seed=0,
    amp=True,
    total_batch_size=768,
    num_iterations=500000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, static_graph=True
    ),
    ddp_fp16_compression=True,
)
