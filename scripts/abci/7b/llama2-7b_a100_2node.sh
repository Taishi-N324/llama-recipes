#!/bin/bash
#$ -l rt_AF=2
#$ -l h_rt=24:00:00
#$ -j y
#$ -o outputs/7b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /groups/gaf51217/fujii/finetune/llama-recipes
source .venv/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
NUM_NODES=$NHOSTS
NUM_GPU_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line
do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done < "$SGE_JOB_HOSTLIST" > "$HOSTFILE_NAME"


# debugging flag
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

CHECKPOINTS_PATH=/groups/gaf51217/fujii/finetune/llama-recipes/checkpoints/llama-2-7b

mkdir -p $CHECKPOINTS_PATH

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python llama_finetuning.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --pure_bf16 \
  --model_name /groups/gaf51217/fujii/finetune/llama2/Llama-2-7b-hf \
  --batch_size_training 1 \
  --dist_checkpoint_root_folder $CHECKPOINTS_PATH \
  --dist_checkpoint_folder fine-tuned \
  --use-mpi
