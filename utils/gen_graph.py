import torch
from torch_geometric.data import Data
import numpy as np

def generate_synthetic_graph(num_nodes=300,
                              feature_dim=32,
                              anomaly_ratio=0.1,
                              inject_structure_noise=True,
                              inject_attribute_noise=True,
                              edge_prob=0.05,
                              seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    
    x = torch.randn((num_nodes, feature_dim))

    
    y = torch.zeros(num_nodes)
    num_anomalies = int(anomaly_ratio * num_nodes)
    anomaly_nodes = np.random.choice(num_nodes, num_anomalies, replace=False)
    y[anomaly_nodes] = 1

    
    if inject_structure_noise:
        extra_edges = []
        for _ in range(num_anomalies * 2):  
            src = np.random.choice(anomaly_nodes)
            dst = np.random.randint(0, num_nodes)
            extra_edges.append([src, dst])
            extra_edges.append([dst, src])
        if extra_edges:
            edge_index = torch.cat(
                [edge_index, torch.tensor(extra_edges, dtype=torch.long).t().contiguous()],
                dim=1
            )

    
    if inject_attribute_noise:
        x[anomaly_nodes] += torch.randn_like(x[anomaly_nodes]) * 5  

    return Data(x=x, edge_index=edge_index, y=y)
