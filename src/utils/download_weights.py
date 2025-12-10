import os
import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"File already exists: {target_path}")
        return

    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print("Download complete.")

def main():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, '..', 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Download SAM Weights
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
    download_file(sam_url, sam_path)

    # 2. Download ObjectClear Weights
    print("Downloading ObjectClear weights from Hugging Face...")
    object_clear_dir = os.path.join(weights_dir, "ObjectClear")
    
    # We only technically *need* the postfuse module safetensors based on the pipeline code,
    # but snapshot_download ensures we have what we need. 
    # The pipeline code specifically looks for `postfuse_module/model.safetensors`
    
    snapshot_download(
        repo_id="jixin0101/ObjectClear",
        local_dir=object_clear_dir,
        local_dir_use_symlinks=False
        # We can filter if we want to save space, but let's grab it all for safety first
        # ignore_patterns=["*.msgpack", "*.h5", "*.ot"] 
    )
    print(f"ObjectClear weights downloaded to {object_clear_dir}")

if __name__ == "__main__":
    main()
