# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (  # noqa: F401
    FullyShardedDataParallel as FSDP,  # type: ignore
    StateDictType,  # type: ignore
    FullStateDictConfig,  # type:ignore : general model non-sharded, non-flattened params
    LocalStateDictConfig,  # type: ignore : flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)
from torch.distributed._shard.checkpoint import (  # noqa: F401
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (  # noqa: F401
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist

from llama_recipes.configs import train_config
from typing import Type, Any

from llama_recipes.configs.fsdp import fsdp_config


def get_date_of_run() -> str:
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run: str = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}", flush=True)
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank: int, cfg: Type[train_config]) -> tuple[int, int, int]:
    folder_name = (
        cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print("No sharded_state_dict checkpoint directory found...skipping")
        return 0, 0, 0

    if rank == 0:
        print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print("checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])

    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")

    epoch: int = checkpoint.get("epoch", 0)
    iteration: int = checkpoint.get("iteration", 0)
    consumed_tokens: int = checkpoint.get("consumed_tokens", 0)
    return epoch, iteration, consumed_tokens


def save_model_and_optimizer_sharded(
    model,
    rank: int,
    cfg: Type[train_config],
    optim=None,
    epoch=None,
    iteration=None,
    consumed_tokens=None,
) -> None:
    """
    save model and optimizer via sharded_state_dict to save_dir
    """

    folder_name: str = (
        cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict: dict[str, Any] = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)
        if epoch is not None:
            state_dict["epoch"] = epoch
        if iteration is not None:
            state_dict["iteration"] = iteration
        if consumed_tokens is not None:
            state_dict["consumed_tokens"] = consumed_tokens

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}, iteration={iteration}")
        print(f"Checkpoint Time = {t1-t0:.4f}\n")


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
) -> None:
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print("--> saving model ...")
        # create save path
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def load_model_checkpoint(model, rank, cfg) -> None:
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print("model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        opt_save_full_path = save_dir / opt_save_name

        print("--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank: int) -> None:
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(  # noqa: F841
        full_optim_state_dict=full_osd, model=model
    )

    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path: str):
    state_dict = {"model": model.state_dict()}

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def save_checkpoint(
    model,
    optimizer,
    train_config: Type[train_config],
    fsdp_config: Type[fsdp_config],
    rank: int,
    epoch: int,
    iteration: int,
) -> None:
    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:  # type: ignore
        save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
        if not train_config.use_peft and train_config.save_optimizer:
            save_optimizer_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
            print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
            print("=====================================================")

    # ABCI Llama-2 Continual Learning use below
    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:  # type: ignore
        if train_config.save_optimizer:
            print(f" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT: rank: {rank}")
            print("=====================================================")
            save_model_and_optimizer_sharded(
                model=model, rank=rank, cfg=train_config, optim=optimizer, iteration=iteration
            )
            print(f"saved model and optimizer checkpoint for epoch {epoch} and iteration {iteration}")
            print("=====================================================")
        else:
            print(f" Saving the FSDP model checkpoints using SHARDED_STATE_DICT: rank: {rank}")
            print("=====================================================")
            save_model_and_optimizer_sharded(model=model, rank=rank, cfg=train_config)  # type: ignore
            print("=====================================================")
