"""Dataset download and iteration for nanoLLaDA. Uses the same ClimbMix dataset as nanoChat."""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from nanollada.common import get_base_dir

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet') and not f.endswith('.tmp')])
    return [os.path.join(data_dir, f) for f in parquet_files]

def parquets_iter_batched(split, start=0, step=1):
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No dataset parquet files found. Run: python -m nanollada.dataset -n 8"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            yield rg.column('text').to_pylist()

def download_single_file(index):
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try: os.remove(path)
                    except: pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)
    successful = sum(1 for s in results if s)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
