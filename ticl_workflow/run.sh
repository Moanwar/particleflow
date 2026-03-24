#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source /afs/cern.ch/work/m/moanwar/private/mlpf/mlpf_env/bin/activate

cd /afs/cern.ch/work/m/moanwar/private/mlpf/particleflow

export PYTHONPATH=$(pwd)
export KERAS_BACKEND=torch
export TFDS_DATA_DIR=/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/tensorflow_datasets

python mlpf/pipeline.py \
  --config /afs/cern.ch/work/m/moanwar/private/mlpf/test_workflow/my_training.yaml \
  --data-dir $TFDS_DATA_DIR \
  --prefix MLPF_ticl_test_ \
  train \
  --gpus 0 \
  --num-steps 10 \
  --dtype float32 \
  --conv-type attention \
  --attention-type math \
  --num-convs 1 \
  --num-workers 1 --prefetch-factor 1
