import pandas as pd
import os

from tqdm import tqdm

SAVE_DIR = 'data/alpaca'
os.makedirs(SAVE_DIR, exist_ok=True)

data = pd.read_parquet('data/train-00000-of-00001-a09b74b3ef9c3b56.parquet')
texts = list(data.text)
print(len(texts))

new_file_counter = 0
for i, text in tqdm(enumerate(texts), total=len(texts)):
    if  i % 5200 == 0:
        new_file_counter += 1
    if new_file_counter == 11:
        # Don't save the last text file/shard. It's too small.
        break
    with open(f"{SAVE_DIR}/_{str(new_file_counter)}.txt", 'a') as f:
        f.write(text+'\n')