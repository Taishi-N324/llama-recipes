#!/bin/bash
#$ -l rt_AF=32
#$ -l h_rt=50:00:00:00
#$ -j y
#$ -o outputs/13B/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

echo "GPU NODE TYPE: ${NODE_TYPE}, NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"

NUM_NODES=$NHOSTS
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

# training settings
NUM_EPOCHS=1

# batch size
BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))

if (($GRADIENT_ACCUMULATION_STEPS < 1)); then
  echo "Error: Gradient Accumulation Steps is less than 1. Exiting."
  exit 1
fi

# optimizer
LR=1e-4
LR_MIN=1e-5
LR_DECAY=0.8
LR_WARMUP=0.027777777777777776
LR_DECAY_STYLE="cosine"
WEIGHT_DECAY=0.1

# seed
SEED=42

# dataset
NUM_WORKERS_DATALOADER=2

# hugginface setting
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/

# checkpoint path
CHECKPOINTS_PATH=/bb/llm/gaf51275/llama/checkpoints/llama-2-13b-performance/llama2-13b_${NODE_TYPE}_${NHOSTS}_FSDP_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}/only_cc
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
  --model_name /bb/llm/gaf51275/jalm/Llama-2-13b-chat-merged-tokenizer-hf \
  --tokenizer_name /bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model \
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
  --dataset "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/tokenized_streaming/only_cc" \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --save_interval_iteration 500 \
  --save_checkpoint_path $CHECKPOINTS_PATH \
  --use_mpi \
  --use_fast_kernels \
  --streaming_datasets_train_path /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/tokenized_streaming/only_cc/ \
  --wandb_name "llama2-13b_${NODE_TYPE}_${NHOSTS}_FSDP_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}"
