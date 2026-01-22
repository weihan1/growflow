import subprocess
import sys
from pathlib import Path

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

install("huggingface-hub")
target_dir = Path("./results")
target_dir.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="weihan1/growflow_checkpoints",
    local_dir=str(target_dir),
    resume_download=True
)

print(f"Checkpoints downloaded to {target_dir.resolve()}")
