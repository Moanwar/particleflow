#export TFDS_DATA_DIR=/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/data/tensorflow_datasets
#export TFDS_DATA_DIR=/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/tensorflow_datasets
#tfds build particleGun_nopu_phase2.py --manual_dir=/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/mlpf/data/cms/raw/ --config=1
#tfds build particleGun_nopu_phase2.py --manual_dir=/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/pikl_files  --config=1

# Set where TFDS goes
export TFDS_DATA_DIR=/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/tensorflow_datasets

# Build all 10 splits from whatever PKL files exist in the directory
for config in 1 2 3 4 5 6 7 8 9 10; do
    tfds build ticl_nopu.py \
        --manual_dir=/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/mix_part_0pu/pikl_files \
        --config=$config
done
