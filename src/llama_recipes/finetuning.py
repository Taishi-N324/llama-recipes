# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import logging
import random

import fire
import numpy as np
import torch
import torch.distributed as torch_distributed
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # type: ignore
import torch.optim as optim
import wandb
import typing
import deepspeed  # noqa: F401
from peft import get_peft_model, prepare_model_for_int8_training  # type: ignore
from pkg_resources import packaging  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config, train_config
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    update_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.distributed import print_rank_0
from llama_recipes.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    get_policies,
    print_model_size,
    setup,
    setup_environ_flags,
    train,
)
from llama_recipes.optimizer import WarmupCosineAnnealingLR
from llama_recipes.utils.sequence_length_warmup import (  # noqa: F401
    SequenceLengthWarmupDistributedSampler,  # noqa: F401
    SequenceLengthWarmupDataset,  # noqa: F401
    CustomDistributedSampler,
)
# from streaming import StreamingDataset
# from streaming import StreamingDataLoader
# from llama_recipes.utils.streaming_dataset_utils import combined_collate_fn
import json
import sentencepiece as spm

def main(**kwargs) -> None:
    # logging 設定
    logging.basicConfig(level=logging.WARNING)

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)  # type: ignore

    # Set the seeds for reproducibility
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # possibly unbound error を解決するために
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    # Distributed args.
    if train_config.use_mpi:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
        local_world_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]
        for var in env_vars:
            if var in os.environ:
                print(f"{var} is defined and its value is: {os.environ[var]}")
            else:
                print(f"{var} is not defined.")
    
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        print(local_rank)
        rank = int(os.environ["RANK"])
        print(rank)
        world_size = int(os.environ["WORLD_SIZE"])
        print(world_size)

    # wandb setting
    if train_config.wandb_name is not None and rank == 0:
        import datetime
        from llama_recipes.utils.wandb_utils import set_config

        wandb_configs: dict[str, typing.Any] = {}
        set_config(wandb_configs=wandb_configs)

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": "ontocord",
            "project": "code-llama",
            "name": train_config.wandb_name,
            "config": wandb_configs,
            "mode" : 'offline',
        }
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(local_rank)  # type: ignore
        clear_gpu_cache(local_rank)  # type: ignore
        setup_environ_flags(rank)  # type: ignore

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception(
                "latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                "please install latest nightly."
            )
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)  # type: ignore
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)  # type: ignore

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)  # type: ignore

    # Load the tokenizer ABCIのLLMでは、paddingはしません
    # hfならこれ
    tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
    # tokenizer = spm.SentencePieceProcessor()
    # tokenizer.Load(train_config.tokenizer_name)

    if train_config.use_peft:
        print(f"Using PEFT method: {train_config.peft_method}", flush=True)
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)  # type: ignore
        model.print_trainable_parameters()

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            print_rank_0("NOTE: freeze transformer layers")
            freeze_transformer_layers(model=model, num_layer=train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,  # type: ignore
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)  # type: ignore
            if train_config.low_cpu_fsdp and rank != 0
            else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")  # type: ignore

    if train_config.use_streaming_datasets:
        pass
    else:

        dataset_config = generate_dataset_config(train_config, kwargs)

        # Load and preprocess the dataset for training and validation
        dataset_train = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="train",
        )

        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Training Set Length = {len(dataset_train)}")  # type: ignore

        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")  # type: ignore

        """
        estimated_total_iterations: 学習にかかる iteration数
        lr_warmup_iterations: learning rateがwarmupしきるのにかかるiterations
        lr_decay_iterations: learning rate が cosineで落ちきるのにかかるiterations
        """
        estimated_total_iterations: int = (
            train_config.num_epochs
            * len(dataset_train)  # type: ignore
            // (train_config.batch_size_training * world_size * train_config.gradient_accumulation_steps)
        )
        lr_warmup_iterations: int = int(estimated_total_iterations * train_config.lr_warmup)
        lr_decay_iterations: int = int(estimated_total_iterations * train_config.lr_decay)

        dataset_length: int = len(dataset_train)  # type: ignore
        if rank == 0:
            print(f"dataset_train: {dataset_length}")  # type: ignore

    # streaming-mosaicmlの組み込み
    if train_config.use_streaming_datasets:
        # UnboundLocalError 回避
        train_sampler = None
        val_sampler = None
        if rank == 0:
            print("train_config.streaming_datasets_train_path",train_config.streaming_datasets_train_path)
        dataset_train = StreamingDataset(local=train_config.streaming_datasets_train_path, split=None, shuffle=True, shuffle_seed=42)
        train_dataloader = StreamingDataLoader(
            dataset_train,
            batch_size=train_config.batch_size_training,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=lambda b: combined_collate_fn(b, max_seq_len=train_config.sequence_length),
        )

        # 1. Check if the path is None
        latest_streaming_datasets_checkpoint_path = os.path.join(train_config.load_checkpoint_path, "latest_streaming_info.json")
        assert latest_streaming_datasets_checkpoint_path is not None, "Path specification is required!"

        # 2. Create an empty file if the file doesn't exist
        if not os.path.exists(latest_streaming_datasets_checkpoint_path):
            save_streaming_datasets_checkpoint_path = os.path.join(train_config.save_checkpoint_path, "latest_streaming_info.json")
            try:
                with open(save_streaming_datasets_checkpoint_path, 'w') as f:
                    pass
            except PermissionError:
                raise PermissionError(f"Could not create file at {save_streaming_datasets_checkpoint_path} due to permission issues!") from None

        # 3. Check the keys in the file
        else:
            with open(latest_streaming_datasets_checkpoint_path, "r") as file:
                content = file.read()
                if content:  # Only process if the file is not empty
                    loaded_dict = json.loads(content)
                    
                    # Check for the existence of keys
                    keys_to_check = ["epoch", "sample_in_epoch", "num_canonical_nodes", "shuffle_seed"]  
                    for key in keys_to_check:
                        assert key in loaded_dict, f"Key {key} not found in the loaded dictionary!"
                    print(f"state_dict_streaming load info {loaded_dict} ")
                        
                    train_dataloader.load_state_dict(loaded_dict)

        dataset_val = StreamingDataset(local=train_config.streaming_datasets_val_path, split=None, shuffle=True)
        eval_dataloader = StreamingDataLoader(
            dataset_val,
            batch_size=train_config.batch_size_training,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=lambda b: combined_collate_fn(b, max_seq_len=train_config.sequence_length),
        )
        if rank == 0:
            print("train_config.num_epochs",train_config.num_epochs)
            print("len(train_dataloader)",len(dataset_train))
            print("train_config.batch_size_training",train_config.batch_size_training)
            print("train_config.gradient_accumulation_steps",train_config.gradient_accumulation_steps)
            print("world_size",world_size)
        # estimated_total_iterations: int = (
        #     train_config.num_epochs
        #     * 100000  # type: ignore
        #     // (train_config.batch_size_training * world_size * train_config.gradient_accumulation_steps)
        # )
        estimated_total_iterations = train_config.estimated_total_iterations
        lr_warmup_iterations: int = int(estimated_total_iterations * train_config.lr_warmup)
        lr_decay_iterations: int = int(estimated_total_iterations * train_config.lr_decay)

        dataset_length: int = 36463349  # type: ignore

        if rank == 0:
            print(f"dataset_train: {dataset_length}")  # type: ignore

    # デフォルト実装
    else:
        train_sampler = None
        val_sampler = None
        if train_config.enable_fsdp:
            train_sampler = CustomDistributedSampler(
                dataset_train,
                rank=torch_distributed.get_rank(),
                num_replicas=torch_distributed.get_world_size(),
                shuffle=True,
                seed=train_config.seed,
            )
            if train_config.run_validation:
                val_sampler = DistributedSampler(
                    dataset_val,
                    rank=torch_distributed.get_rank(),
                    num_replicas=torch_distributed.get_world_size(),
                    seed=train_config.seed,
                )

        # Create DataLoaders for the training and validation dataset
        # NOTE: we need to set worker_init_fn to set seed for each worker
        def worker_init_fn(worker_id: int) -> None:
            worker_seed = train_config.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        train_dataloader: DataLoader = DataLoader(
            dataset=dataset_train,
            batch_size=train_config.batch_size_training,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=train_sampler if train_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
            worker_init_fn=worker_init_fn,
        )

        eval_dataloader: typing.Optional[DataLoader] = None
        if train_config.run_validation:
            eval_dataloader = DataLoader(
                dataset_val,
                batch_size=train_config.val_batch_size,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                sampler=val_sampler if val_sampler else None,
                drop_last=True,
                collate_fn=default_data_collator,
            )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),  # type: ignore
            lr=train_config.lr,
            betas=train_config.adamw_betas,
            eps=train_config.adamw_eps,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),  # type: ignore
            lr=train_config.lr,
            betas=train_config.adamw_betas,
            eps=train_config.adamw_eps,
            weight_decay=train_config.weight_decay,
        )
    
    # wandb config update
    if train_config.wandb_name is not None and rank == 0:
        # iteration info
        wandb.config.update(
            {
                "total_iteration": estimated_total_iterations,
                "warmup_iteration": lr_warmup_iterations,
                "decay_iteration": lr_decay_iterations,
            }
        )

    if train_config.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=lr_warmup_iterations,
            decay_iterations=lr_decay_iterations,
            max_iterations=estimated_total_iterations,
            eta_min=train_config.lr_min,
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    

    # Start the training process
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        sampler=train_sampler,  # type: ignore
        tokenizer=tokenizer,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        train_config=train_config,
        fsdp_config=fsdp_config if train_config.enable_fsdp else None,
        local_rank=local_rank if train_config.enable_fsdp else None,
        rank=rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
