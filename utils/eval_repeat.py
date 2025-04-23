import torch
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from pygod.metric import eval_f1
from torch.nn import functional as F

activation_map = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
}

def eval_repeat(
    model_class,
    dataset,
    binary_label,
    params,
    n_runs=10,
    epochs=200,
    device="cuda",
    save_path=None
):
    aucs, aps, f1s = [], [], []

    if isinstance(device, str) and device.startswith("cuda"):
        gpu_id = torch.device(device).index if torch.device(device).index is not None else 0
    elif isinstance(device, int):
        gpu_id = device
        device = f"cuda:{gpu_id}"
    else:
        gpu_id = -1  

    for run in range(n_runs):
        print(f"\n[Run {run+1}/{n_runs}]")

        if torch.cuda.is_available() and gpu_id >= 0:
            mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
            mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3
            print(f"[GPU {gpu_id} Memory] allocated: {mem_alloc:.2f} GiB, reserved: {mem_reserved:.2f} GiB")

        this_run_params = dict(params)

        if "activation" in this_run_params:
            act_str = this_run_params.pop("activation")
            if act_str not in activation_map:
                raise ValueError(f"Unsupported activation function: {act_str}")
            this_run_params["act"] = activation_map[act_str]

        this_run_params.pop("gpu", None)

        model = model_class(
            **this_run_params,
            epoch=epochs,
            gpu=gpu_id,
            verbose=0
        )

        model.fit(dataset)
        pred, score = model.predict(return_score=True)

        label_np = binary_label.cpu().numpy()
        score_np = score.cpu().detach().numpy()
        auc = roc_auc_score(label_np, score_np)
        ap = average_precision_score(label_np, score_np)
        f1 = eval_f1(binary_label.cpu(), pred.cpu())

        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    summary = {
        "AUC-ROC (mean±std)": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        "AP (mean±std)": f"{np.mean(aps):.4f} ± {np.std(aps):.4f}",
        "F1 (mean±std)": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
    }

    results_df = pd.DataFrame({"AUC": aucs, "AP": aps, "F1": f1s})

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

    return summary, results_df



