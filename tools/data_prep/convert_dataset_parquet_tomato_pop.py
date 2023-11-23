"""Convert LAION-POP parquet shards to mosaic streaming dataset.

How is the data stored?
The text are stored as string and the images are stored in list before being pickled.

How to set the concat number?
Set CONCAT_NUM = ?

How to run this file?
At the path where you want to create data, python run this file.

"""

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
import pandas as pd

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

# Read the parquet file
data_dir = "/p/fastdata/mmlaion/m3_laion_pop"
shard_dir = os.path.join(data_dir, "shards")
parquet_files = [f for f in os.listdir(shard_dir) if f.endswith('.parquet')]
# data = pd.read_parquet(os.path.join(shard_dir, "00000000.parquet"))

columns = {'text': 'str', 'images': 'pkl'}
output_path = './pop_data'

# Check if the path exists
if os.path.exists(output_path):
    shutil.rmtree(output_path)  # Remove the directory and all its contents

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
success_row = 0
total_data = 0

CONCAT_NUM = 3 # concat 10 samples into 1.

with MDSWriter(columns=columns, out=output_path, progress_bar=1) as out:
    
    for parquet_file in tqdm(parquet_files):
        data = pd.read_parquet(os.path.join(shard_dir, parquet_file))
        data = data[data['jpg'].notna()]
        total_data += data.shape[0]
    
        for i in range(0, len(data), CONCAT_NUM):  # Process in batches of 10
            batch = data.iloc[i:min(i+CONCAT_NUM, len(data))]

            # Concatenate captions and collect images
            # concatenated_text = '<|Image|>'+'<|Image|>'.join(batch['caption'].fillna(batch['alt_txt']).tolist())
            concatenated_text = ' This is a placeholder sentence. ' * 10000
            image_list = batch['jpg'].tolist()

            if image_list:
                image_pickle = pickle.dumps(image_list)
                sample = {'text': concatenated_text, 'images': image_pickle}
                out.write(sample)
        break # process only one parquet for quick experiment