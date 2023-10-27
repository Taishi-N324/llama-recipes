#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=00:30:00
#$ -j y
#$ -o outputs/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes/
source .env/bin/activate

### YOUR HUGGINGFACE TOKEN ###
export HF_TOKEN=
export HF_HOME=/home/acf15834ji/.cache/huggingface

# inference
# --hf-model-path には、huggingfaceのモデルnameまたは、学習したモデルのpathを指定する
# --hf-tokenizer-path についても同様

python tools/inference/inference.py \
  --hf-model-path /bb/llm/gaf51275/llama/hf_checkpoints-4500iter \
  --hf-tokenizer-path /bb/llm/gaf51275/llama/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb_16k_aligned_8/merged_tokenizer_hf \
  --hf-token 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 \
  --hf-cache-dir $HF_HOME \
  --input-text "日本の首都は"
