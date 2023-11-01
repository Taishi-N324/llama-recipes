#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster
#SBATCH --output=%j_0_log.out  # change this line to your output file 

cd /p/home/jusers/nakamura2/juwels/nakamura2/ABCI-llama-recipes
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


# training settings
NUM_EPOCHS=50


# optimizer
LR=1e-4
LR_MIN=1e-5
LR_DECAY=0.80
LR_WARMUP=0.05
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY=0.1

# seed
SEED=42

# dataset
NUM_WORKERS_DATALOADER=2

# checkpoint path


# checkpoint path
CHECKPOINTS_PATH=/p/home/jusers/nakamura2/juwels/nakamura2/ABCI-llama-recipes/checkpoints/
mkdir -p $CHECKPOINTS_PATH

NUM_GPU_PER_NODE=4

NUM_NODES=$NHOSTS
NUM_GPUS=$((${SLURM_NNODES} * ${NUM_GPU_PER_NODE}))


# batch size
BATCH_SIZE=2
GLOBAL_BATCH_SIZE=1024
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))

if (($GRADIENT_ACCUMULATION_STEPS < 1)); then
  echo "Error: Gradient Accumulation Steps is less than 1. Exiting."
  exit 1
fi



mkdir -p ./hostfile

# HOSTFILE_NAME=./hostfile/hostfile_${SLURM_JOB_ID}
# while read -r line
# do
#   echo "${line} slots=${NUM_GPU_PER_NODE}"
# done < "$SLURM_JOB_NODELIST" > "$HOSTFILE_NAME"

HOSTFILE_NAME=./hostfile/hostfile_${SLURM_JOB_ID}

scontrol show hostnames $SLURM_JOB_NODELIST | while read -r line
do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done > "$HOSTFILE_NAME"


mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python examples/finetuning.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --peft_method None \
  --mixed_precision \
  --pure_bf16 \
  --num_epochs $NUM_EPOCHS \
  --model_name  /p/home/jusers/nakamura2/juwels/.cache/huggingface/hub/models--codellama--CodeLlama-34b-hf/snapshots/fda69408949a7c6689a3cf7e93e632b8e70bb8ad \
  --tokenizer_name /p/home/jusers/nakamura2/juwels/.cache/huggingface/hub/models--codellama--CodeLlama-34b-hf/snapshots/fda69408949a7c6689a3cf7e93e632b8e70bb8ad  \
  --batch_size_training $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --lr $LR \
  --lr_min $LR_MIN \
  --lr_warmup $LR_WARMUP \
  --lr_decay $LR_DECAY \
  --lr_decay_style $LR_DECAY_STYLE \
  --weight_decay $WEIGHT_DECAY \
  --fsdp_activation_checkpointing \
  --seed $SEED \
  --dataset "samsum_dataset" \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --save_interval_iteration 500 \
  --save_checkpoint_path $CHECKPOINTS_PATH \
  --use_mpi \
  --use_fast_kernels \
  --streaming_datasets_train_path  /p/project/ccstdl/nakamura2/ABCI-llama-recipes/sample_datasets \
  --wandb_name "llama2-7b_${NODE_TYPE}_SLURM_NNODES_${SLURM_NNODES}_FSDP_NUM_GPUS_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}" \
  --estimated_total_iterations 17500
done

wait