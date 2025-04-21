import torch
import optuna
import json
import hashlib
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import functional as F

activation_map = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
}

def sanitize_params(params):
    clean_params = dict(params)
    if "activation" in clean_params:
        act_str = clean_params.pop("activation")
        if act_str not in activation_map:
            raise ValueError(f"Unsupported activation: {act_str}")
        clean_params["act"] = activation_map[act_str]
    return clean_params

def save_best_params(model_class, dataset_name, best_params, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_class.__name__.lower()
    path = os.path.join(output_dir, f"best_params_{model_name}_{dataset_name}.json")
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params to {path}")

def tune_model_with_optuna(
    model_class,
    dataset,
    label_transform_fn,
    param_space_fn,
    n_trials=30,
    metric="auc",
    train_epochs=30,
    device="cuda",
    return_all_metrics=False,
    show_progress=True,
    dataset_name=None
):
    binary_label = label_transform_fn(dataset.y)

    def objective(trial):
        try:
            params = param_space_fn(trial)

            if "activation" in params:
                act_str = params.pop("activation")
                if act_str not in activation_map:
                    raise ValueError(f"Unsupported activation: {act_str}")
                params["act"] = activation_map[act_str]

            if isinstance(device, int):
                use_gpu = device
            elif isinstance(device, str) and device.startswith("cuda"):
                use_gpu = 0 if torch.cuda.is_available() else -1
            else:
                use_gpu = -1

            model = model_class(
                **params,
                epoch=train_epochs,
                gpu=use_gpu,
                verbose=0
            )

            model.fit(dataset)
            _, score = model.predict(return_score=True)

            label_np = binary_label.cpu().numpy()
            score_np = score.cpu().detach().numpy()

            auc = roc_auc_score(label_np, score_np)
            ap = average_precision_score(label_np, score_np)

            if return_all_metrics:
                trial.set_user_attr("auc", auc)
                trial.set_user_attr("ap", ap)

            if metric == "auc":
                return auc
            elif metric == "ap":
                return ap
            else:
                raise ValueError(f"Unsupported metric '{metric}'")

        except Exception as e:
            print(f"[Trial {trial.number} Failed] {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    best_params = study.best_params
    best_params_clean = sanitize_params(best_params)

    hash_str = hashlib.md5(str(best_params_clean).encode()).hexdigest()[:6]
    model_name = model_class.__name__.lower()
    print(f"[{model_name}-{dataset_name}] Best AUC: {study.best_value:.4f} | Hash: {hash_str}")

    if dataset_name is not None:
        save_best_params(model_class, dataset_name, best_params)

    return study


