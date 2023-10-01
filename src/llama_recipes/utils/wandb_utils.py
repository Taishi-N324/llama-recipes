from llama_recipes.configs import fsdp_config, train_config


def set_config(wandb_configs: dict) -> None:
    # train_config
    wandb_configs["model_name"] = train_config.model_name
    wandb_configs["enable_fsdp"] = train_config.enable_fsdp
    wandb_configs["low_cpu_fsdp"] = train_config.low_cpu_fsdp
    wandb_configs["run_validation"] = train_config.run_validation
    wandb_configs["batch_size_training"] = train_config.batch_size_training
    wandb_configs["gradient_accumulation_steps"] = train_config.gradient_accumulation_steps
    wandb_configs["num_epochs"] = train_config.num_epochs
    wandb_configs["num_workers_dataloader"] = train_config.num_workers_dataloader
    wandb_configs["lr"] = train_config.lr
    wandb_configs["weight_decay"] = train_config.weight_decay
    wandb_configs["gamma"] = train_config.gamma
    wandb_configs["seed"] = train_config.seed
    wandb_configs["use_fp16"] = train_config.use_fp16
    wandb_configs["mixed_precision"] = train_config.mixed_precision
    wandb_configs["val_batch_size"] = train_config.val_batch_size
    wandb_configs["dataset"] = train_config.dataset
    wandb_configs["peft_method"] = train_config.peft_method
    wandb_configs["use_peft"] = train_config.use_peft
    wandb_configs["freeze_layers"] = train_config.freeze_layers
    wandb_configs["num_freeze_layers"] = train_config.num_freeze_layers
    wandb_configs["quantization"] = train_config.quantization
    wandb_configs["one_gpu"] = train_config.one_gpu
    wandb_configs["save_model"] = train_config.save_model
    wandb_configs["save_optimizer"] = train_config.save_optimizer
    wandb_configs["use_fast_kernels"] = train_config.use_fast_kernels
    wandb_configs["use_mpi"] = train_config.use_mpi

    # fsdp_config
    wandb_configs["mixed_precision"] = fsdp_config.mixed_precision
    wandb_configs["use_fp16"] = fsdp_config.use_fp16
    wandb_configs["sharding_strategy"] = fsdp_config.sharding_strategy
    wandb_configs["checkpoint_type"] = fsdp_config.checkpoint_type
    wandb_configs["fsdp_activation_checkpointing"] = fsdp_config.fsdp_activation_checkpointing
    wandb_configs["pure_bf16"] = fsdp_config.pure_bf16
    wandb_configs["optimizer"] = fsdp_config.optimizer
