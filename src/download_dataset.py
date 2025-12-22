import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from urllib.request import urlretrieve

from src.config import IMAGES_DIR, MASKS_DIR, VECTOR_DIR

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
csv_path = os.path.join(VECTOR_DIR, "ai4boundaries_ftp_urls_all.csv")
df = pd.read_csv(csv_path)

print("Total records in CSV:", len(df))

filtered_df = (
    df[df["file_id"].str.startswith("NL")]
    .sort_values(by="file_id", ascending=True)
    .head(50)          # 🔴 change this later (100)
    .copy()
)

print(filtered_df["file_id"])

# --------------------------------------------------
# Ensure output directories exist
# --------------------------------------------------
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

# --------------------------------------------------
# Download function
# --------------------------------------------------
def download_tiles(df):
    for _, row in df.iterrows():
        field = row["file_id"]

        img_url = row["sentinel2_images_file_url"]
        mask_url = row["sentinel2_masks_file_url"]

        img_path = os.path.join(IMAGES_DIR, f"{field}.nc")
        mask_path = os.path.join(MASKS_DIR, f"{field}.tif")

        print(f"\nDownloading field: {field}")

        try:
            if not os.path.exists(img_path):
                urlretrieve(img_url, img_path)
                print("  ✓ Image downloaded")
            else:
                print("  ↪ Image already exists")

            if not os.path.exists(mask_path):
                urlretrieve(mask_url, mask_path)
                print("  ✓ Mask downloaded")
            else:
                print("  ↪ Mask already exists")

        except Exception as e:
            print(f"  ✗ Error downloading {field}: {e}")

# --------------------------------------------------
# Run
# --------------------------------------------------
download_tiles(filtered_df)