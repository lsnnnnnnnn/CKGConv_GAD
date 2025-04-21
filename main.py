import argparse
import torch
import json
from pathlib import Path
from pygod.detector import DOMINANT
from utils.metric_wrapper import MetricWrapper


def load_custom_data(name: str, rrwp: bool, device: str):
    root = Path("/data1/shengen/STATS403/project/data")
    subdir = "real" if "inj_" in name or "books" in name else "synthetic"
    fname = f"{name}{'_rrwp' if rrwp else ''}.pt"
    path = root / subdir / fname
    data = torch.load(path, map_location=device)
    print(f"📦 Loaded: {path}")
    return data.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='inj_cora')
    parser.add_argument('--rrwp', action='store_true', help='use _rrwp.pt file')
    parser.add_argument('--metric', type=str, nargs='+', default=['f1', 'accuracy', 'auroc'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()

    # Load custom dataset
    data = load_custom_data(args.dataset, rrwp=args.rrwp, device=args.device)

    # Train model
    model = DOMINANT(hid_dim=64, num_layers=4)
    model.fit(data)

    # Predict scores
    outlier_scores = torch.as_tensor(outlier_scores, device=args.device)
    ground_truth   = torch.as_tensor(ground_truth,   device=args.device)

    #if not isinstance(outlier_scores, torch.Tensor):
        #outlier_scores = torch.tensor(outlier_scores, device=args.device)
    #if not isinstance(ground_truth, torch.Tensor):
        #ground_truth = torch.tensor(ground_truth, device=args.device)

    # Evaluate and store results
    results = {"dataset": args.dataset, "rrwp": args.rrwp}
    for metric_name in args.metric:
        kwargs = {}
        if metric_name in ["f1", "accuracy"]:
            kwargs['task'] = 'binary'
        metric = MetricWrapper(metric_name, **kwargs)
        score = metric(outlier_scores, ground_truth)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"\n✅ Results saved to: {output_path}\n")


if __name__ == '__main__':
    main()

