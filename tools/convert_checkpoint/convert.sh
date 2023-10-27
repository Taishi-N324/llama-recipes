#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=00:30:00
#$ -j y
#$ -o outputs/
#$ -cwd
set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes/
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

LLAMA_PRETRAINED_PATH=$1
CHECKPONT_ITER_PATH=$2
HF_SAVE_PATH=$3
TOKENIZER_PATH=$4

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python tools/convert_checkpoint/convert.py \
  --llama_pretrained_path $LLAMA_PRETRAINED_PATH \
  --checkpont_iter_path $CHECKPONT_ITER_PATH \
  --hf_save_path $HF_SAVE_PATH

cp -r $TOKENIZER_PATH/* $HF_SAVE_PATH