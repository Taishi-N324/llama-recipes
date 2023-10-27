#!/bin/bash

# Default paths
DEFAULT_LLAMA_PRETRAINED_PATH="/bb/llm/gaf51275/jalm/modified-llama2-chat-aligned-8/llama2-7b-chat-merged-tokenizer-16k-hf"
# DEFAULT_BASE_DIR="/bb/llm/gaf51275/llama/checkpoints/llama-2-7b-performance/llama2-7b_a100_4_FSDP_32_GLOBAL_BATCH_SIZE_1024/taishi-data-shuffle-all_maxlr1e-4_min3.3-e4"
# DEFAULT_BASE_DIR="/bb/llm/gaf51275/llama/checkpoints/llama-2-7b-performance/llama2-7b_a100_2_FSDP_16_GLOBAL_BATCH_SIZE_1024/taishi-data-LrMin1e-5-LrMax3e-5-shuffle-full"

DEFAULT_BASE_DIR="/bb/llm/gaf51275/llama/checkpoints/llama-2-7b-performance/llama2-7b_a100_8_FSDP_64_GLOBAL_BATCH_SIZE_1024/llama-2-7b-gbs_1024_16k_merge_ja90_en10_lrmax1e-4_min3.3e-6"
#DEFAULT_BASE_DIR="/bb/llm/gaf51275/llama/checkpoints/llama-2-7b-performance/llama2-7b_a100_4_FSDP_32_GLOBAL_BATCH_SIZE_1024/taishi-data-shuffle-all_maxlr1e-4_min3.3-e6"
DEFAULT_HF_SAVE_PATH="/bb/llm/gaf51275/llama/from_fsdp_hf_checkpoints/llama2-7b-8node-gbs1024-taishi-data-shuffle-en10-ja90-maxlr1e-4_min3.3e-6"
DEFAULT_TOKENIZER_PATH="/bb/llm/gaf51275/llama/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_16k_aligned_8/merged_tokenizer_hf"

# Overwrite default paths with arguments if provided
LLAMA_PRETRAINED_PATH=${1:-$DEFAULT_LLAMA_PRETRAINED_PATH}
CHECKPONT_ITER_PATH=${2:-$DEFAULT_BASE_DIR}
HF_SAVE_PATH=${3:-$DEFAULT_HF_SAVE_PATH}
TOKENIZER_PATH=${4:-$DEFAULT_TOKENIZER_PATH}

# Loop to submit jobs
# for i in $(seq 0 500 2000); do
for i in $(seq 500 500 2500); do
    formatted_i=$(printf "%07d" $i)
    qsub -ar 26132 -g gaf51275 convert.sh $LLAMA_PRETRAINED_PATH $CHECKPONT_ITER_PATH/iter_${formatted_i} $HF_SAVE_PATH/iter_${formatted_i} $TOKENIZER_PATH
done
