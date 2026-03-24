#!/bin/bash
export PYTHONPATH=$(pwd)
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3


python scripts/evaluate.py --config configs/eval_zero_shot_classification.py \
    --checkpoint-path /path/to/your/ckpt \
    --train-config configs/train_uncha_vit_b.py



