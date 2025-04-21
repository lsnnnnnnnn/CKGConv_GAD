import os
import torch
from utils.rrwp import add_full_rrwp_new  
from torch_geometric.data import Data


DATA_DIR = "/data1/shengen/STATS403/project/data"
SUBDIRS = ["real", "synthetic"]

def add_rrwp_and_save(file_path):
    print(f"Processing {file_path}...")
    data = torch.load(file_path)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

    data = add_full_rrwp_new(data, walk_length=8)

    save_path = file_path.replace(".pt", "_rrwp.pt")
    torch.save(data.cpu(), save_path)
    print(f"Saved with RRWP at {save_path}")

def main():
    for subdir in SUBDIRS:
        dir_path = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".pt") and not fname.endswith("_rrwp.pt"):
                file_path = os.path.join(dir_path, fname)
                try:
                    add_rrwp_and_save(file_path)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
