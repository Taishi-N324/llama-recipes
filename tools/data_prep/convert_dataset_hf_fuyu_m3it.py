"""Convert M3IT parquet shards."""

from transformers import FuyuProcessor, TomatoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import base64
from PIL import Image
from io import BytesIO
from streaming import MDSWriter
import pickle
import os
import shutil
from tqdm.auto import tqdm  # Use tqdm.auto for a progress bar that works well in notebooks and terminals
import torch
import numpy as np

def load_image(image, size=None, center_crop=False):
    if center_crop:
        image = image.crop(
            (
                (image.width - min(image.width, image.height)) // 2,
                (image.height - min(image.width, image.height)) // 2,
                (image.width + min(image.width, image.height)) // 2,
                (image.height + min(image.width, image.height)) // 2,
            )
        )
    if size is not None:
        image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0).float()
    image = image / 127.5 - 1.0
    return image

ds_name = "image-paragraph-captioning"
dataset = load_dataset("MMInstruction/M3IT", ds_name)

columns = {'text': 'str', 'image_base64_str': 'str'}
output_path = './m3it_mds_train'

# Check if the path exists
if os.path.exists(output_path):
    shutil.rmtree(output_path)  # Remove the directory and all its contents

dataloader = DataLoader(dataset["train"], batch_size=1)

model_id = "/p/scratch/ccstdl/transformers_cache/tomato-1113"
processor = TomatoProcessor.from_pretrained(model_id)

# Function to convert base64 strings to PIL images
def convert_b64_image(base64_list):
    images = []
    if isinstance(base64_list, str):
        base64_list = [base64_list]
    for b in base64_list:
        # Decode base64 and open image
        image = Image.open(BytesIO(base64.b64decode(b)))
        # Convert to 'RGB' if not already 3 channels
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
    return images

def convert_utf8_to_str(utf8_list):
    return [x.encode('utf-8') for x in utf8_list]

# Create the output directory
os.makedirs(output_path, exist_ok=True)

error_count = 0
# Use MDSWriter to write samples
with MDSWriter(columns=columns, out=output_path, progress_bar=1) as out:
    for batch in tqdm(dataset['train'], desc='Converting batches'):
        # Concatenate the string fields to form 'full_prompt'
        # batch['full_prompt'] = [x + y + z for x, y, z in zip(batch['instruction'], batch['inputs'], batch['outputs'])]
        batch['full_prompt'] = batch['instruction']+' '+batch['inputs']+' '+batch['outputs']
        batch['full_prompt'] = batch['full_prompt'] * 100
        
        images = convert_b64_image(batch['image_base64_str'][0])        
        inputs = {
            'text': batch['full_prompt'],
            'image_base64_str': batch['image_base64_str'][0],
        }
        # Serialize the processed inputs
        sample = inputs
        
        # Write the sample to the output
        out.write(sample)
