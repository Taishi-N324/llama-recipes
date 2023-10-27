#!/bin/bash

set -e

# BASE_PATH="/bb/llm/gaf51275/llama/from_fsdp_hf_checkpoints/llama2-7b-4node-gbs1024-taishi-data-shuffle-en50-ja50-maxlr1e-4_min3.3e-5"
#BASE_PATH="/bb/llm/gaf51275/llama/from_fsdp_hf_checkpoints/llama2-7b-4node-gbs1024-taishi-data-shuffle-en50-ja50-maxlr1e-4_min3.3e-6"
# BASE_PATH="/bb/llm/gaf51275/llama/from_fsdp_hf_checkpoints/llama2-7b-4node-gbs1024-taishi-data-shuffle-en50-ja50-maxlr3e-5_min1e-5"
BASE_PATH="/bb/llm/gaf51275/llama/from_fsdp_hf_checkpoints/llama2-7b-8node-gbs1024-taishi-data-shuffle-en10-ja90-maxlr1e-4_min3.3e-6"
# MODEL_NAME="llama2-7b-4node-gbs1024-taishi-data-shuffle-en50-ja50-maxlr1e-4_min3.3e-5-vocab-16k"
MODEL_NAME="llama2-7b-8node-gbs1024-taishi-data-shuffle-en10-ja90-maxlr1e-4_min3.3e-6-vocab-16k"

for ITER in $(seq 3000 500 4000); do
    ITER_FORMATTED=$(printf "iter_%07d" $ITER)
    nohup python upload.py \
        $BASE_PATH/$ITER_FORMATTED \
        tokyotech-llm/$MODEL_NAME-$ITER_FORMATTED \
        main > llama2-7b-8node-gbs1024-taishi-data-shuffle-en10-ja90-maxlr1e-4_min3.3e-6-vocab-16k_$ITER_FORMATTED.log 2>&1 &
done
