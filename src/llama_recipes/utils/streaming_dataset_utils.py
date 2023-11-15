from typing import Dict, Any, List
import torch
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from transformers import TomatoProcessor

def _read_binary_tokenized_sample(sample: Dict[str, Any], max_seq_len: int) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(sample['tokens'],
                      dtype=np.int64)[:max_seq_len].copy())

def intermediate_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    return {'input_ids': torch.stack([_read_binary_tokenized_sample(sample, max_seq_len) for sample in batch])}

def combined_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    intermediate_result = intermediate_collate_fn(batch, max_seq_len)

    # input_idsからattention_maskを計算 今回事前にトークナイズする時はpadされてないと思うが念の為
    attention_mask = (intermediate_result['input_ids'] != 0).long()


    labels = intermediate_result['input_ids'].clone()

    result = {
        'input_ids': intermediate_result['input_ids'],
        'attention_mask': attention_mask,
        'labels': labels
    }
    return result

processor = TomatoProcessor.from_pretrained('/p/scratch/ccstdl/transformers_cache/tomato-1113')

def convert_b64_image(base64_list):
    images = []
    for b in base64_list:
        # Decode base64 and open image
        image = Image.open(BytesIO(base64.b64decode(b)))
        # Convert to 'RGB' if not already 3 channels
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
    return images

def fuyu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # convert list of dict to dict of list
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    result = processor(text=batch['text'], images=convert_b64_image(batch['image_base64_str']), return_tensors="pt", padding='max_length', truncation=True, max_length=15000)
    for key, value in result.items():
        print(f"Key: {key}")
        print(f"Content: {value}")
        try:
            print(f"Shape: {value.shape}")
        except:
            print("List")

    return result