import torch
import os


def get_dataset_path(dataset_name, base_dir="/data1/shengen/STATS403/project/data"):
    """
    Automatically find the correct .pt file from real/ or synthetic/ folder.
    """
    for subdir in ["real", "synthetic"]:
        path = os.path.join(base_dir, subdir, dataset_name + ".pt")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Dataset {dataset_name}.pt not found in {base_dir}/real or synthetic")


def load_dataset(dataset_name, use_rrwp=False, base_dir="/data1/shengen/STATS403/project/data"):
    """
    Load the dataset. If use_rrwp is True, look for *_rrwp.pt and assign rrwp_val + rrwp_index.
    Otherwise, load normal graph without edge_attr.
    """
    file_path = get_dataset_path(dataset_name, base_dir)
    data = torch.load(file_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    if use_rrwp:
        if hasattr(data, 'rrwp_val') and hasattr(data, 'rrwp_index'):
            data.edge_attr = data.rrwp_val.to(device)
            data.edge_index = data.rrwp_index.to(device)
        else:
            raise ValueError(f"{dataset_name}.pt does not contain RRWP encoding.")
    else:
        data.edge_attr = None  

    return data