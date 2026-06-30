#!/bin/bash
JOB_INDEX=$1
FILENAME=$2

export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
export SCRAM_ARCH=slc7_amd64_gcc10
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source /afs/cern.ch/work/m/moanwar/private/mlpf/mlpf_env/bin/activate

echo "Starting job on $(date)"
echo "Running on: $(uname -a)"
echo "JOB_INDEX=${JOB_INDEX}"

FILES_PER_JOB=5
START=$(( JOB_INDEX * FILES_PER_JOB + 1 ))
END=$(( START + FILES_PER_JOB - 1 ))
echo "Processing lines ${START} to ${END}"

# Extract the files for this job into a temporary input list
TMPLIST="input_job_${JOB_INDEX}.txt"
sed -n "${START},${END}p" "${FILENAME}" > "${TMPLIST}"

if [ ! -s "${TMPLIST}" ]; then
    echo "No files found for job index ${JOB_INDEX}, exiting."
    exit 0
fi

cat "${TMPLIST}"

OUTPUT="ticl_zll_nopu_${JOB_INDEX}.pkl"

python3 postprocessing_ticl_ttbar_nopu.py \
    --input "${TMPLIST}" \
    --output "${OUTPUT}" \
    --events-per-pkl 5000

if [ $? -ne 0 ]; then
    echo "ERROR: postprocessing failed"
    exit 1
fi

# Copy output to EOS
for f in ticl_zll_nopu_${JOB_INDEX}_*.pkl; do
    if [ -f "$f" ]; then
        xrdcp "$f" root://eosuser.cern.ch//eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/ttbar_0pu_v3/pikle_filesv1_1pTmin/
        if [ $? -eq 0 ]; then
            echo "xrdcp succeeded for $f, removing."
            rm -f "$f"
        else
            echo "xrdcp failed for $f, keeping."
        fi
    fi
done

rm -f "${TMPLIST}"
echo "Job ${JOB_INDEX} done on $(date)"
