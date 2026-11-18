"""Download the Brain Tumor MRI dataset from Kaggle into ./data (about 157 MB)."""
from pathlib import Path

import kagglehub

out = Path(__file__).parent / "data"
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", output_dir=str(out))
print("Dataset in:", path)
