#!/bin/bash
unset PYTHONPATH
source /home/moanwar/mlpf/twoD-env/bin/activate
cd /home/moanwar/mlpf/particleflow/
export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch
export TFDS_DATA_DIR=/cms/data/store/user/moanwar/mlpf_data/ttbar_0pu/tensorflow_datasets_1minpT_ttbar_zll_qcd
export CUDA_VISIBLE_DEVICES=0
python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-ticl-v1 \
  --production-name cms_ticl \
  --data-dir $TFDS_DATA_DIR \
  --experiment-dir /cms/data/store/user/moanwar/mlpf_data/experiments_heptv2_ttbar_zll_qcd/ \
  train \
  --gpus 1 \
  --num_steps 200000 \
  --dtype bfloat16 \
  --num_workers 4 \
  --prefetch_factor 4
