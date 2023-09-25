#!/bin/bash
#$ -l rt_F=16
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
cd /home/acf15649kv/work/finetune/llama-recipes
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

# learning config
NUM_EPOCHS=10

BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$((${BATCH_SIZE} * ${NUM_GPUS} * ${GRADIENT_ACCUMULATION_STEPS}))

LR=1e-4
WEIGHT_DECAY=1e-2
SEED=42

NUM_WORKERS_DATALOADER=2

# checkpoint path
CHECKPOINTS_PATH=/groups/gaf51217/fujii/checkpoints/llama-2-7b/llama-recipies
mkdir -p $CHECKPOINTS_PATH

# hugginface setting
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/

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
  --use_fp16 \
  --num_epochs $NUM_EPOCHS \
  --model_name /groups/gaf51217/fujii/finetune/llama2/Llama-2-7b-hf \
  --batch_size_training $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --lr $LR \
  --weight_decay $WEIGHT_DECAY \
  --fsdp_activation_checkpointing \
  --seed $SEED \
  --num_workers_dataloader $NUM_WORKERS_DATALOADER \
  --save_model \
  --save_optimizer \
  --dist_checkpoint_root_folder $CHECKPOINTS_PATH \
  --dist_checkpoint_folder "${NODE_TYPE}_${NHOSTS}_FSDP_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}" \
  --use_mpi \
  --use_fast_kernels \
  --wandb_name "llama2-7b_${NODE_TYPE}_${NHOSTS}_FSDP_${NUM_GPUS}_GLOBAL_BATCH_SIZE_${GLOBAL_BATCH_SIZE}"
