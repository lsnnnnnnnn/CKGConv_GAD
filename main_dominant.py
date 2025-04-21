import torch
from torch_geometric.data import Data
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc, eval_f1
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_paths = [
    "/data1/shengen/STATS403/project/data/real/books.pt",
    "/data1/shengen/STATS403/project/data/real/books.pt",
    "/data1/shengen/STATS403/project/data/synthetic/gen_500.pt",
    "/data1/shengen/STATS403/project/data/synthetic/structural_anomaly_graph.pt"
]

results = []

for path in dataset_paths:
    data = torch.load(path).to(device)

    model = DOMINANT(
        hid_dim=64,
        num_layers=4,
        epoch=100,
        lr=0.004,
        contamination=0.1,
        gpu=0 if device.type == 'cuda' else -1
    )
    model.fit(data)

    pred, score = model.predict(data, return_score=True)

    auc = eval_roc_auc(data.y, score)
    f1 = eval_f1(data.y, pred)
    acc = (pred == data.y).sum().item() / data.num_nodes

    results.append({
        'dataset': os.path.basename(path),
        'auc_roc': auc,
        'accuracy': acc,
        'f1_score': f1
    })


df = pd.DataFrame(results)
df.to_csv("dominant_results.csv", index=False)

print("Results saved to dominant_results.csv")
