#!/bin/bash
export PYTHONPATH=$(pwd)
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0,1,2,3

./scripts/train.sh --config configs/train_uncha_vit_b.py --num-gpus 4  --output-dir ./train_results/test --checkpoint-period 10000 
