#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o process_log/
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

cd tools/data_prep

# python merge_json_data_mult.py --dirs \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_0 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_1 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_2 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_3 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_4 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_5 \
# /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_6 \
# --dest /bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/merged_train_0_6/train

base_path="/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja"
dirs=""
for i in {0..6}; do
    dirs="${dirs} ${base_path}/merged_train_${i}"
done

python merge_json_data_mult.py --dirs $dirs --dest ${base_path}/merged_train_0_6_check/train
