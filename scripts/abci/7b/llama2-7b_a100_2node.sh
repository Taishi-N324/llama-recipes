#!/bin/bash

# swich virtual env
cd /home/taishi/llama-recipes
source venv/bin/activate
cd /home/taishi/ABCI-llama-recipes

# distributed settings
export MASTER_ADDR=10.130.184.10
export MASTER_PORT=12800

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

# if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
#   export NUM_GPU_PER_NODE=4
#   NODE_TYPE="v100"
# elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
#   export NUM_GPU_PER_NODE=8
NODE_TYPE="a100"
export NUM_GPU_PER_NODE=8
# else
#   echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
# fi

# NUM_NODES=$NHOSTS
# NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

# mkdir -p ./hostfile

# HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
# while read -r line
# do
#   echo "${line} slots=${NUM_GPU_PER_NODE}"
# done < "$SGE_JOB_HOSTLIST" > "$HOSTFILE_NAME"

HOSTFILE_NAME=/home/taishi/ABCI-llama-recipes/hostfile/4node

# debugging flag
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# training settings
NUM_EPOCHS=1
NUM_GPUS=8
# batch size
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))

if (($GRADIENT_ACCUMULATION_STEPS < 1)); then
  echo "Error: Gradient Accumulation Steps is less than 1. Exiting."
  exit 1
fi

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
CHECKPOINTS_PATH=/model/taishi/checkpoints/checkpoints/llama-2-7b-gbs_${GLOBAL_BATCH_SIZE}-${NODE_TYPE}_${NHOSTS}
mkdir -p $CHECKPOINTS_PATH



# checkpoint path
CHECKPOINTS_PATH=/model/taishi/checkpoints/7b/llama-2-7b-gbs_1024_48k_merge_ja10_en90
mkdir -p $CHECKPOINTS_PATH


# run
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
  --model_name  /home/taishi/models/llama2-7b-chat-merged-tokenizer-48k-hf/llama2-7b-chat-merged-tokenizer-48k-hf \
  --tokenizer_name jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_48k_aligned_8/merged_tokenizer_sp//jalm_llama.model  \
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
  --dataset "/model/taishi/datasets/sp_48k/merge_ja10_en90/shuffle1/" \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --save_interval_iteration 500 \
  --save_checkpoint_path $CHECKPOINTS_PATH \
  --use_mpi \
  --use_fast_kernels \
  --streaming_datasets_train_path  /model/taishi/datasets/sp_48k/merge_ja10_en90/shuffle1/ \
  --wandb_name "llama2-7b_${NODE_TYPE}_${NHOSTS}_FSDP_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}" \
  --estimated_total_iterations 17500
