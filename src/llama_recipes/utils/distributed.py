from torch import distributed as torch_distributed


def is_rank_0() -> bool:
    return torch_distributed.is_initialized() and torch_distributed.get_rank() == 0


def print_rank_0(message) -> None:
    if torch_distributed.is_initialized() and torch_distributed.get_rank() == 0:
        print(message, flush=True)
