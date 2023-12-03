#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%j_0_log.out  # change this line to your output file 
set -e

cd /p/scratch/ccstdl/xu17/liangyu/ly_recipes
source /p/project/ccstdl/nakamura2/miniconda3/bin/activate /p/project/ccstdl/nakamura2/llama-recipe-torch2.1_cuda-11.8

# Network Configuration
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

echo $SLURM_JOB_GPUS
echo $SLURM_NTASKS
echo $SLURM_NODELIST

export WANDB_RUN_ID=$SLURM_JOB_ID

# Convert SLURM_JOB_GPUS to an array
IFS=',' read -ra GPU_ARRAY <<< "$SLURM_JOB_GPUS"

# Get the number of GPUs from the length of the array
NUM_GPUS=${#GPU_ARRAY[@]}

export TOTAL_GPUS=$(($NUM_GPUS * $SLURM_NTASKS))
echo $TOTAL_GPUS

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"

export MASTER_ADDR=$master_addr
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_PORT=12802
export COUNT_NODE=$SLURM_NNODES

# Print System Information
echo "GPUs available to job: $SLURM_JOB_GPUS"
echo "Total tasks: $SLURM_NTASKS"

NUM_GPU_PER_NODE=4
NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# PRETRAINED_PATH=$1
# CHECKPONT_ITER_PATH=$2
# HF_SAVE_PATH=$3
# TOKENIZER_PATH=$4


PRETRAINED_PATH=/p/scratch/ccstdl/transformers_cache/tomato-1113
MODEL_NAME=tomato
CHECKPONT_ITER_PATH=/p/scratch/ccstdl/xu17/liangyu/ly_recipes/checkpoints/iter_0000500
HF_SAVE_PATH=/p/scratch/ccstdl/xu17/liangyu/ly_recipes/checkpoints_hf/iter_0000500


mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python tools/convert_checkpoint/convert.py \
  --pretrained_path $PRETRAINED_PATH \
  --model_name $MODEL_NAME \
  --checkpont_iter_path $CHECKPONT_ITER_PATH \
  --hf_save_path $HF_SAVE_PATH