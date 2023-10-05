from typing import List
import datasets

from transformers import LlamaTokenizer
from tqdm.contrib.concurrent import process_map

from itertools import chain


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size: int = chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {k: v + list(chain(*batch[k])) for k, v in self.residual.items()}

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [v[i: i + self.chunk_size] for i in range(0, chunk_num * self.chunk_size, self.chunk_size)]
                for k, v in concatenated_samples.items()
            }
            self.residual = {k: v[(chunk_num * self.chunk_size):] for k, v in concatenated_samples.items()}
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


def load_dataset(split: str, tokenizer, return_dict, paths: List[str]) -> None:
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
        path="json",
        data_files=paths,
        num_proc=2,
    )

    def tokenize_map(sample):
        return tokenizer(sample["text"])

    dataset = (
        raw_dataset["train"]
        .map(
            process_map(tokenize_map, raw_dataset["train"], max_workers=4),
            batched=True,
            remove_columns=list(raw_dataset["train"].features),
        )
        .map(Concatenator(chunk_size=4096), batched=True)
    )
    return_dict[split] = dataset[split]


# def get_llm_jp_dataset_multiprocessed(tokenizer):
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()

#     num_processes = 30
#     processes = []

#     # Trainデータのパスを分割
#     train_paths = [
#         f"/bb/llm/gaf51275/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_train_{i}.jsonl" for i in range(38)
#     ] + ["/bb/llm/gaf51275/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_train_0.jsonl"]

#     chunk_size: int = len(train_paths) // num_processes
#     print(f"chunk_size: {chunk_size}", flush=True)
#     for i in range(num_processes):
#         paths_chunk: list[str] = train_paths[i * chunk_size: (i + 1) * chunk_size]
#         p = multiprocessing.Process(target=load_dataset, args=("train", tokenizer, return_dict, paths_chunk))
#         processes.append(p)
#         p.start()

#     # Validationデータは一つのプロセスで読み込む
#     val_paths: list[str] = [
#         "/bb/llm/gaf51275/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_val_0.jsonl",
#         "/bb/llm/gaf51275/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_val_0.jsonl",
#     ]
#     p_val = multiprocessing.Process(target=load_dataset, args=("test", tokenizer, return_dict, val_paths))
#     processes.append(p_val)
#     p_val.start()

#     for p in processes:
#         p.join()

#     # プロセス間の結果を統合
#     combined_dataset = {}
#     combined_dataset["train"] = sum((return_dict[f"train_{i}"] for i in range(num_processes)), [])
#     combined_dataset["test"] = return_dict["test"]
#     return combined_dataset

def get_llm_jp_dataset(tokenizer, split: str = "train"):
    if split == "train":
        print("dataset_paths call")
        train_path = "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_train_0.jsonl"
        print("train_path call")
        dataset_paths: list[str] = [
            train_path
        ]

        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=2,
        )
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
            )
            .map(Concatenator(chunk_size=4096), batched=True)
        )
        return dataset["train"]
    else:
        dataset_paths: list[str] = [
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_val_0.jsonl",
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_val_0.jsonl",
        ]
        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=2,
        )
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
            )
            .map(Concatenator(chunk_size=4096), batched=True)
        )
        return dataset["test"]


tokenizer = LlamaTokenizer.from_pretrained(
    "/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_clueweb/merged_tokenizer_hf"
)
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)

# この関数を呼び出してマルチプロセスでデータセットを取得は動かないので
# get_llm_jp_dataset_multiprocessed(tokenizer)
get_llm_jp_dataset(tokenizer)
