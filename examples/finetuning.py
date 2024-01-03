# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import sys
sys.path.append("/bb/llm/gaf51275/llm-jp/taishi-work-space/llama-recipes/src")

import fire
from llama_recipes.finetuning import main
import streaming

# def clean_stale_memory():
#     try:
#         streaming.base.util.clean_stale_shared_memory()
#     except FileNotFoundError as e:
#         print(f"Warning: {e}")

def clean_stale_memory():
    try:
        streaming.base.util.clean_stale_shared_memory()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    except FileExistsError as e:
        print(f"Error: Conflicting file state detected. Details: {e}")
    except PermissionError as e:
        print(f"Error: Permission denied. Details: {e}")
    except MemoryError as e:
        print(f"Error: Memory error occurred. Details: {e}")
    except Exception as e:
        # Catch-all for other exceptions
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    clean_stale_memory()
    fire.Fire(main)
