#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
from huggingface_hub import snapshot_download
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)
print("Downloading dataset from Hugging Face...")
snapshot_download(
    repo_id="weihan1/growflow_data", 
    repo_type="dataset",
    local_dir=str(data_dir),
    resume_download=True
)
print("Download complete.\n")
for archive in data_dir.glob("*.tar.zst"):
    print(f"Extracting {archive.name} ...")
    subprocess.run(
        f"zstd -dc {archive} | tar -xf -",
        shell=True,
        check=True
    )
print("Finished extracting archives.\n")
for folder_name in ["captured", "synthetic"]:
    src = Path(folder_name)
    dst = data_dir / folder_name
    if src.exists():
        print(f"Moving {src} -> {dst}")
        shutil.move(str(src), str(dst))

print("\nAll dataset folders are now in ./data")
