# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional


@dataclass
class train_config:
    model_name: str = "/home/acf15834ji/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
    tokenizer_name: str = "/home/acf15834ji/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = False
    batch_size_training: int = 4
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    num_epochs: int = 10
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    lr_min: float = 1e-5
    lr_decay: float = 0.80  # ratio of decay
    lr_warmup: float = 0.002  # ratio of warmup
    lr_decay_style: str = "cosine"
    use_sequence_length_schedule: bool = False
    sequence_length: int = 4096
    sequence_length_warmup_min: int = 8
    sequence_length_warmup: float = 0.15
    weight_decay: float = 0.1
    gamma: float = 0.85
    adamw_eps: float = 1e-5
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset: str = "ja_wikipedia_dataset"
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    save_checkpoint_path: str = ""
    save_optimizer: bool = True  # will be used if using FSDP
    load_checkpoint_path: str = ""
    save_interval_iteration: int = 100
    use_fast_kernels: bool = True  # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_mpi: bool = False
    use_streaming_datasets: bool = True
    streaming_datasets_train_path: str = "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/streaming_ja/ja_wiki_no_zstd/train/"
    streaming_datasets_val_path: str = "/bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes/tools/data_prep/samsum2"
    latest_streaming_datasets_checkpoint_path: str = "/bb/llm/gaf51275/llama/taishi-work-streaming/ABCI-llama-recipes/scripts/abci/13b/latest.json"
    wandb_name: Optional[str] = None
