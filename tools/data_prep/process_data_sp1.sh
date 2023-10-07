#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=48:00:00
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

tokenizer_path="/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_sp/jalm_llama.model"

for i in {5..9}; do
    input_path="/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_train_${i}.jsonl"
    output_path="/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja_sp/13b/ja_cc_train_${i}/"
    
    echo "input_path=${input_path}"
    echo "output_path=${output_path}"
    echo "tokenizer_path=${tokenizer_path}"

    python convert_dataset_json_sp.py \
      --path $input_path \
      --out_root $output_path \
      --split train \
      --concat_tokens 4096 --tokenizer $tokenizer_path
done

# node GPU上での処理は不要なはず
# --compression zstdはtmpファイル周りでエラーを起こすのでやめたほうがいいかもしれない
# 圧縮したい場合は引数として取るようにする
# sp用