#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=4:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# swich virtual env
cd /bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes
source .env/bin/activate

# pip version up
pip install --upgrade pip

# install pytorch (nightly)
pip install --pre torch==2.1.0.dev20230905+cu118 --index-url https://download.pytorch.org/whl/nightly/cu118

# pip install requirements
pip install -r requirements.txt

# distirbuted training requirements
pip install mpi4py
pip install deepspeed

# huggingface requirements
pip install huggingface_hub

# install flash-atten
pip install ninja packaging wheel
pip install flash-attn==2.3.0 --no-build-isolation

# install mosaicml-streaming
pip install mosaicml-streaming

# # timm を install しようとすると、torch 2.0.1を強制的にinstallさせてくるので注意

# 昨日自分が追加でインストールしたもの 最新のflash attention 2.3.1はエラーが起こる 
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# pip install flash-attn==2.3.0