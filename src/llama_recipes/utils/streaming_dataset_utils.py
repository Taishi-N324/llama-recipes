from typing import Dict, Any, List
import torch
import numpy as np

def _read_binary_tokenized_sample(sample: Dict[str, Any], max_seq_len: int) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(sample['tokens'],
                      dtype=np.int64)[:max_seq_len].copy())

def intermediate_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    return {'input_ids': torch.stack([_read_binary_tokenized_sample(sample, max_seq_len) for sample in batch])}

def combined_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    intermediate_result = intermediate_collate_fn(batch, max_seq_len)

    # input_idsからattention_maskを計算 TODO llm-jpをまねる
    attention_mask = (intermediate_result['input_ids'] != 0).long()

    labels = intermediate_result['input_ids'].clone()

    result = {
        'input_ids': intermediate_result['input_ids'],
        'attention_mask': attention_mask,
        'labels': labels
    }
    return result