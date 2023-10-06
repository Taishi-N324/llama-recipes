# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from pathlib import Path
from pkg_resources import packaging  # type: ignore
from contextlib import nullcontext

import torch
import torch.cuda.nccl as nccl
from torch import distributed as torch_distributed  # noqa: F401
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.utils import clip_grad_norm_  # type: ignore
from tqdm import tqdm
from transformers import LlamaTokenizer
from llama_recipes.configs.fsdp import fsdp_config
from llama_recipes.configs.training import train_config

from llama_recipes.model_checkpointing import save_checkpoint
from llama_recipes.model_checkpointing.checkpoint_handler import load_model_sharded
from llama_recipes.policies import fpSixteen, bfSixteen_mixed, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace

from typing import Optional, Type, Any
import wandb

from llama_recipes.utils.sequence_length_warmup import SequenceLengthWarmupDistributedSampler


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)


def train(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    sampler: SequenceLengthWarmupDistributedSampler | DistributedSampler,
    tokenizer,
    optimizer: torch.optim.AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_accumulation_steps: int,
    train_config: Type[train_config],
    fsdp_config: Optional[Type[fsdp_config]] = None,
    local_rank: Optional[int] = None,
    rank: Optional[int] = None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps:
            The number of steps to accumulate gradients before performing
            a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predictions

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # possibly unbound error 解消のため default値を設定
    world_size: int = 1
    local_rank = local_rank if local_rank is not None else 0

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()  # type: ignore
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext  # type: ignore

    train_prep: list[torch.Tensor] = []  # train perplexity
    train_loss: list[float] = []  # train loss
    val_prep: list[torch.Tensor] = []  # validation perplexity
    val_loss: list[float] = []  # validation loss
    epoch_times: list[float] = []
    checkpoint_times: list[float] = []
    results: dict[str, Any] = {}
    best_val_loss = float("inf")

    # set model info
    if rank == 0 and train_config.wandb_name:
        model_config: dict[str, Any] = {}
        model_config["activation_function"] = model.config.hidden_act
        model_config["hidden_size"] = model.config.hidden_size
        model_config["model_type"] = model.config.model_type
        model_config["max_position_embeddings"] = model.config.max_position_embeddings
        model_config["num_attention_heads"] = model.config.num_attention_heads
        model_config["num_hidden_layers"] = model.config.num_hidden_layers
        model_config["vocab_size"] = model.config.vocab_size
        model_config["model_architecture"] = model.config.architectures[0]

        print(f"model info: {model}")
        print(f"model config: {model.config}")
        wandb.config.update(model_config)

        # distributed training info
        wandb.config.update({"world_size": world_size})

    if train_config.use_sequence_length_schedule:
        pass

    last_epoch: int = 0
    last_iteration: int = 0
    consumed_tokens: int = 0
    # model load & 学習状態を復元
    if train_config.load_checkpoint_path != "":
        if train_config.enable_fsdp:
            torch_distributed.barrier()

        last_epoch, last_iteration, consumed_tokens = load_model_sharded(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            rank=rank if rank is not None else 0,
            cfg=train_config,
        )
        # sampler state load
        load_dir: str = train_config.load_checkpoint_path

        if not Path(load_dir).exists():
            if rank == 0:
                print("No checkpoint directory found...skipping")
        else:
            from llama_recipes.model_checkpointing.checkpoint_handler import read_latest_value

            try:
                last_iteration: int = read_latest_value(f"{load_dir}/latest")
                sampler_checkpoint_path: str = (
                    load_dir + "/sampler/iter_{:07d}/sampler_checkpoint.pt".format(last_iteration)
                )
                sampler.load_state_dict(torch.load(sampler_checkpoint_path))  # type: ignore
            except FileNotFoundError or ValueError:
                if rank == 0:
                    print("No latest iteration file found")

        if train_config.enable_fsdp:
            torch_distributed.barrier()

    wandb_iteration: int = 0
    for epoch in range(last_epoch, train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        iteration_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss: float = 0.0
            total_length: int = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch}",
                total=total_length,
                disable=(rank != 0),
            )

            accumulation_loss: float = 0.0
            # checkpointをloadした場合は、次のステップから始める
            next_step: int = (
                last_iteration * gradient_accumulation_steps if last_iteration != 0 else 0
            )
            for step, batch in enumerate(train_dataloader, start=next_step):
                wandb_iteration = (
                    epoch * len(train_dataloader) + step // gradient_accumulation_steps
                )
                if train_config.use_sequence_length_schedule:
                    # sequence length warmup
                    current_seq_len: int = min(4096, max(64, 64 + 4 * wandb_iteration))
                    batch = {key: value[:, :current_seq_len] for key, value in batch.items()}

                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to("cuda:0")

                with autocast():
                    loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()

                # accumulate loss for logging
                if rank == 0 and train_config.wandb_name:
                    accumulation_loss += loss.detach().float()

                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()  # type: ignore
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        scaler.step(optimizer)  # type: ignore (suppress ubound error)
                        scaler.update()  # type: ignore (suppress ubound error)
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        pbar.update(step // gradient_accumulation_steps)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                        pbar.update(step // gradient_accumulation_steps)

                # gradient clipping
                if train_config.clip_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), train_config.clip_grad_norm)

                pbar.set_description(
                    f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float() * gradient_accumulation_steps}, lr: {optimizer.param_groups[0]['lr']:.6f}, accumulation_step: {step % gradient_accumulation_steps + 1}/{gradient_accumulation_steps}, iteration: {wandb_iteration})"  # noqa: E501
                )

                if (
                    rank == 0
                    and train_config.wandb_name
                    and (step + 1) % gradient_accumulation_steps == 0
                ):
                    # gradient accumulation stepsごとにwandbにログを送る
                    wandb_stats: dict[str, Any] = {}

                    # training info
                    wandb_stats["training/loss"] = accumulation_loss
                    wandb_stats["training/perplexity"] = torch.exp(
                        torch.tensor(accumulation_loss).clone().detach()
                    )
                    accumulation_loss = 0.0

                    # utils info
                    batch_size: int = batch["input_ids"].shape[0]
                    sequence_length: int = batch["input_ids"].shape[1]

                    wandb_stats["utils/batch_size"] = batch_size
                    wandb_stats["utils/global_batch_size"] = (
                        batch_size * world_size * gradient_accumulation_steps
                    )
                    wandb_stats["utils/seq_len"] = sequence_length
                    wandb_stats["utils/gradient_accumulation_steps"] = gradient_accumulation_steps
                    wandb_stats["utils/epoch"] = epoch
                    wandb_stats["utils/step"] = step

                    # optimizer info
                    wandb_stats["optimizer/lr"] = optimizer.param_groups[0]["lr"]

                    optimizer_states_1: list[float] = [0.0] * 8
                    optimizer_states_2: list[float] = [0.0] * 4

                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            # optimizer state が空の場合は logging しない
                            if not optimizer.state:
                                continue

                            optimizer_states_1[0] += (
                                torch.norm(optimizer.state[param]["exp_avg_sq"]).item()
                            ) ** 2  # type: ignore
                            optimizer_states_1[1] += (
                                torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt()).item()  # type: ignore
                            ) ** 2
                            optimizer_states_1[2] += (
                                torch.norm(optimizer.state[param]["exp_avg"]).item()
                            ) ** 2  # type: ignore
                            optimizer_states_1[3] += (torch.norm(param).item()) ** 2  # type: ignore
                            optimizer_states_1[4] += torch.norm(
                                optimizer.state[param]["exp_avg_sq"], p=1
                            ).item()  # type: ignore
                            optimizer_states_1[5] += torch.norm(
                                optimizer.state[param]["exp_avg_sq"].sqrt(), p=1
                            ).item()  # type: ignore
                            optimizer_states_1[6] += torch.norm(
                                optimizer.state[param]["exp_avg"], p=1
                            ).item()  # type: ignore
                            optimizer_states_1[7] += torch.norm(param, p=1).item()
                            optimizer_states_2[0] = max(
                                optimizer_states_2[0],  # type: ignore
                                abs(optimizer.state[param]["exp_avg_sq"].max().item()),  # type: ignore
                                abs(optimizer.state[param]["exp_avg_sq"].min().item()),  # type: ignore
                            )
                            optimizer_states_2[1] = max(
                                optimizer_states_2[1],
                                optimizer.state[param]["exp_avg_sq"].sqrt().abs_().max().item(),  # type: ignore
                            )
                            optimizer_states_2[2] = max(
                                optimizer_states_2[2],  # type: ignore
                                abs(optimizer.state[param]["exp_avg"].max().item()),  # type: ignore
                                abs(optimizer.state[param]["exp_avg"].min().item()),  # type: ignore
                            )
                            optimizer_states_2[3] = max(
                                optimizer_states_2[3],
                                abs(param.max().item()),  # type: ignore
                                abs(param.min().item()),  # type: ignore
                            )
                    if optimizer.state:  # optimizer stateがない場合はloggingしない
                        # rank:0でしかoptimizer stateをloggingしないので world sizeで割る必要はない
                        wandb_stats["optimizer/variance_l2"] = optimizer_states_1[0] ** 0.5
                        wandb_stats["optimizer/variance_sqrt_l2"] = optimizer_states_1[1] ** 0.5
                        wandb_stats["optimizer/momentum_l2"] = optimizer_states_1[2] ** 0.5
                        wandb_stats["optimizer/weight_l2"] = optimizer_states_1[3] ** 0.5
                        wandb_stats["optimizer/variance_l1"] = optimizer_states_1[4]
                        wandb_stats["optimizer/variance_sqrt_l1"] = optimizer_states_1[5]
                        wandb_stats["optimizer/momentum_l1"] = optimizer_states_1[6]
                        wandb_stats["optimizer/weight_l1"] = optimizer_states_1[7]
                        wandb_stats["optimizer/variance_abs_max"] = optimizer_states_2[0]
                        wandb_stats["optimizer/variance_sqrt_abs_max"] = optimizer_states_2[1]
                        wandb_stats["optimizer/momentum_abs_max"] = optimizer_states_2[2]
                        wandb_stats["optimizer/weight_abs_max"] = optimizer_states_2[3]

                    # stats
                    iteration_elapsed_time = time.perf_counter() - iteration_start_time
                    iteration_start_time = time.perf_counter()

                    tokens_per_sec = (
                        batch_size
                        * sequence_length
                        * gradient_accumulation_steps
                        / iteration_elapsed_time
                        * world_size
                    )
                    wandb_stats["stats/1_iteration_time"] = iteration_elapsed_time
                    wandb_stats["stats/tokens_pef_sec"] = tokens_per_sec
                    wandb_stats["stats/30B_tokens_day"] = (
                        30 * (1000**3) / tokens_per_sec / 60 / 60 / 24
                    )
                    wandb_stats["stats/300B_tokens_day"] = (
                        300 * (1000**3) / tokens_per_sec / 60 / 60 / 24
                    )
                    wandb_stats["stats/1T_tokens_day"] = (
                        (1000**4) / tokens_per_sec / 60 / 60 / 24
                    )
                    wandb_stats["stats/tokens_per_sec_per_gpu"] = tokens_per_sec / world_size

                    checkpoint_activations_factor = 3
                    if (
                        fsdp_config is not None and fsdp_config.fsdp_activation_checkpointing
                    ):  # type ignore
                        checkpoint_activations_factor = 4

                    num_layers: int = model.config.num_hidden_layers
                    hidden_size: int = model.config.hidden_size
                    vocab_size: int = model.config.vocab_size

                    # tflops calculation
                    flops_per_iteration: float = (
                        24
                        * checkpoint_activations_factor
                        * batch_size
                        * sequence_length
                        * gradient_accumulation_steps
                        * num_layers
                        * (hidden_size**2)
                    ) * (
                        1.0
                        + (sequence_length / (6.0 * hidden_size))
                        + (vocab_size / (16.0 * num_layers * hidden_size))
                    )
                    tflops: float = flops_per_iteration / (iteration_elapsed_time * (10**12))
                    wandb_stats["stats/tflops"] = tflops

                    wandb.log(wandb_stats, step=wandb_iteration + 1)

                    print("------------------------------------------------------------------")
                    print(f"iteration: {wandb_iteration + 1} , tflops: {tflops}")
                    print(
                        "------------------------------------------------------------------",
                        flush=True,
                    )

                if (
                    wandb_iteration + 1
                ) % train_config.save_interval_iteration == 0 and not train_config.use_peft:
                    # 全プロセスがcheckpointを保存できるような状態になるまで待つ
                    if train_config.enable_fsdp:
                        torch_distributed.barrier()
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        train_config=train_config,
                        fsdp_config=fsdp_config,  # type: ignore
                        rank=rank if rank is not None else 0,
                        epoch=epoch,
                        iteration=wandb_iteration + 1,
                    )
                    # 全プロセスがcheckpointを保存し終えるまで待つ
                    if train_config.enable_fsdp:
                        torch_distributed.barrier()
                    if rank == 0:
                        # sampler state save
                        load_dir: str = train_config.load_checkpoint_path
                        os.makedirs(
                            load_dir + "/sampler/iter_{:07d}".format(wandb_iteration + 1),
                            exist_ok=True,
                        )
                        sampler_checkpoint_path: str = (
                            load_dir
                            + "/sampler/iter_{:07d}/sampler_checkpoint.pt".format(
                                wandb_iteration + 1
                            )
                        )
                        torch.save(sampler.state_dict(), sampler_checkpoint_path)  # type: ignore

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            torch_distributed.all_reduce(total_loss, op=torch_distributed.ReduceOp.SUM)
        train_epoch_loss: float = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss: float = train_epoch_loss / world_size
        train_perplexity: torch.Tensor = torch.exp(train_epoch_loss)  # type: ignore

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"  # noqa: E501
                )
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
            )

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(
                model=model,
                train_config=train_config,
                eval_dataloader=eval_dataloader,  # type: ignore
                local_rank=local_rank,
                tokenizer=tokenizer,
            )
            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    torch_distributed.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print("we are about to save the PEFT modules")
                    else:
                        print("we are about to save the PEFT modules")
                    model.save_pretrained(train_config.output_dir)
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        train_config=train_config,
                        fsdp_config=fsdp_config,  # type: ignore
                        rank=rank if rank is not None else 0,
                        epoch=epoch,
                        iteration=epoch * len(train_dataloader),
                    )
                if train_config.enable_fsdp:
                    torch_distributed.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"  # noqa: E501
                )
        else:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"  # noqa: E501
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep  # type: ignore (suppress ubound error)
        results["avg_eval_loss"] = avg_eval_loss  # type: ignore (suppress ubound error)
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(
    model,
    train_config: Type[train_config],
    eval_dataloader: DataLoader,
    local_rank: int,
    tokenizer,
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    world_size: int = 1  # suppress ubound error
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss

    with MemoryTrace() as memtrace:  # noqa: F841
        for step, batch in enumerate(
            tqdm(eval_dataloader, colour="green", desc="evaluating Epoch")
        ):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to("cuda:0")
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        torch_distributed.all_reduce(eval_loss, op=torch_distributed.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)  # type: ignore

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer: int) -> None:
    """transformerの一部のlayerをfreezeする

    Args:
        model: モデル
        num_layer (int): freezeするlayerの数 [0〜 num_layer)
    """
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model) -> None:
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    torch_distributed.init_process_group("nccl")


def setup_environ_flags(rank: int) -> None:
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print("--> Running with torch torch_distributed debug set to detail")


def cleanup() -> None:
    """Clean up the process group after training"""
    torch_distributed.destroy_process_group()


def clear_gpu_cache(rank: Optional[int] = None) -> None:
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model) -> dict[Any, Any]:
    """Get the data types of model parameters"""
    parameter_dtypes: dict[Any, Any] = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support: bool = (
        torch.version.cuda  # type: ignore
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)  # type: ignore
        and torch_distributed.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
        else:
            print("bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be helpful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {
        k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")
    }
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith("__")}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (following FSDP checkpointing style) using properties of the train_config object
    folder_name: str = train_config.save_checkpoint_path

    save_dir = Path(folder_name)
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml: str = yaml.dump(train_params_dict, indent=4)
    file_name: str = os.path.join(save_dir, "train_params.yaml")

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, "w") as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")
