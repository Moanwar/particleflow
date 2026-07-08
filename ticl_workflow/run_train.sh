#!/bin/bash
unset PYTHONPATH
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source /home/moanwar/mlpf/twoD-env/bin/activate
cd /home/moanwar/mlpf/particleflow/
export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch
export TFDS_DATA_DIR=/cms/data/store/user/moanwar/mlpf_data/ttbar_0pu/tensorflow_datasets
export CUDA_VISIBLE_DEVICES=0
python mlpf/pipeline.py \
  --config /home/moanwar/mlpf/particleflow/ticl_workflow/my_training.yaml \
  --data-dir $TFDS_DATA_DIR \
  --prefix MLPF_ticl_test_ \
  --experiment-dir /cms/data/store/user/moanwar/mlpf_data/experiments_ttbar_ptcut_v2/ \
  train \
  --gpus 1 \
  --num-steps 200000 \
  --dtype bfloat16 \
  --conv-type attention \
  --attention-type flash \
  --num-convs 3 \
  --num-workers 4 --prefetch-factor 2
