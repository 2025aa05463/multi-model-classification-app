"""
data_utils.py

Utility to download a Kaggle dataset using kagglehub (or local files), extract CSV files, and return a pandas DataFrame and the chosen file path.
"""
import os
import zipfile
import glob
import pandas as pd


def fetch_kaggle_data(dataset="iabhishekofficial/mobile-price-classification", out_dir=None, use_kagglehub=True, prefer_file_names=None):
    """Download and load a CSV from a Kaggle dataset using kagglehub.

    Returns (df, chosen_path).
    Raises Exception on failure.
    """
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "data_kaggle")
    os.makedirs(out_dir, exist_ok=True)

    data_dir = out_dir

    if use_kagglehub:
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset)
            # If kagglehub returns a zip file path, extract it
            if isinstance(path, str) and path.lower().endswith('.zip') and os.path.exists(path):
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall(out_dir)
                data_dir = out_dir
            elif isinstance(path, str) and os.path.isdir(path):
                data_dir = path
            else:
                # unknown return — treat out_dir as data_dir
                data_dir = out_dir
        except Exception as e:
            # re-raise with context
            raise RuntimeError(f"kagglehub dataset_download failed: {e}")

    # look for CSV files (including subdirectories)
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset directory: {data_dir}")

    if prefer_file_names is None:
        prefer_file_names = [
            "mobile_price_training.csv",
            "mobile_price_train.csv",
            "train.csv",
            "train"
        ]

    chosen = None
    for f in csv_files:
        if os.path.basename(f).lower() in prefer_file_names:
            chosen = f
            break
    if chosen is None:
        # fallback: pick the first CSV (sorted)
        chosen = sorted(csv_files)[0]

    df = pd.read_csv(chosen)
    return df, chosen
