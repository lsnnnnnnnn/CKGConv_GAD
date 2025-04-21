import os
import sys
import torch
from torch_geometric.data import Data
from utils.gen_graph import generate_synthetic_graph
from utils.rrwp import add_full_rrwp_new


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = '/data1/shengen/STATS403/project/data/synthetic'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'synthetic_structural.pt')


def generate_structural_anomaly_dataset(num_nodes=500, anomaly_ratio=0.1, feature_dim=16):
    data = generate_synthetic_graph(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        anomaly_ratio=anomaly_ratio,
        inject_structure_noise=True,
        inject_attribute_noise=False,
        seed=42
    )
    return data


def add_rrwp_encoding(data, walk_length=8):
    data = data.to(device)
    if not hasattr(data, 'edge_weight'):
        data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float, device=device)

    import types
    from torch_geometric.graphgym import config
    from torch_geometric.graphgym.config import cfg
    if cfg is None or not hasattr(cfg, 'posenc_RRWP'):
        config.cfg = types.SimpleNamespace()
        config.cfg.posenc_RRWP = {'local_topK': None}

    orig_eye = torch.eye
    def eye_gpu(*args, **kwargs):
        return orig_eye(*args, **kwargs).to(device)
    torch.eye = eye_gpu

    try:
        data = add_full_rrwp_new(data, walk_length=walk_length)
    finally:
        torch.eye = orig_eye

    return data


def ckgconv_structural():
    data = generate_structural_anomaly_dataset()
    data = data.to(device)
    data = add_rrwp_encoding(data)
    return data


if __name__ == "__main__":
    print("Generating synthetic GAD dataset with RRWP on device:", device)
    data = ckgconv_structural()

    data = data.cpu()
    torch.save(data, save_path)
    print(f"Saved dataset to: {save_path}")
    print(f"  x: {data.x.shape}, edge_index: {data.edge_index.shape}, RRWP abs: {data.rrwp.shape}")
    print(f"  y (anomalies): {int(data.y.sum().item())} / {data.num_nodes}")