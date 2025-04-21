import operator as op
from typing import Union, Callable, Optional, Dict, Any
import torch
import warnings

from torchmetrics.functional import (
    accuracy,
    f1_score,
    auroc,
)

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: str = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
        target_to_int: bool = False,
    ):
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.target_to_int = target_to_int

        if isinstance(operator, str):
            op_name = operator.lower()
            if op_name in ["greater", "gt"]:
                self.op_str = ">"
                operator = op.gt
            elif op_name in ["lower", "lt"]:
                self.op_str = "<"
                operator = op.lt
            else:
                raise ValueError(f"operator `{op_name}` not supported")
        elif callable(operator):
            self.op_str = operator.__name__
        else:
            raise TypeError(f"operator must be `str` or `callable`, got: {type(operator)}")

        self.operator = operator

    def compute(self, preds: torch.Tensor, target: torch.Tensor):
        if self.th_on_preds:
            preds = self.operator(preds, self.threshold)
        if self.th_on_target:
            target = self.operator(target, self.threshold)
        if self.target_to_int:
            target = target.to(int)
        return preds, target

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        return self.compute(preds, target)

    def __repr__(self):
        return f"{self.op_str}{self.threshold}"


METRICS_DICT = {
    "accuracy": accuracy,
    "f1": f1_score,
    "auroc": auroc,
}


class MetricWrapper:
    def __init__(
        self,
        metric: Union[str, Callable],
        threshold_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[str, int]] = None,
        cast_to_int: bool = False,
        **kwargs,
    ):
        if isinstance(metric, str):
            assert metric in METRICS_DICT, f"Unsupported metric `{metric}`"
            self.metric = METRICS_DICT[metric]
            self.metric_name = metric
        else:
            self.metric = metric
            self.metric_name = metric.__name__

        if threshold_kwargs is None and self.metric_name in {"f1", "accuracy"}:
            threshold_kwargs = {"threshold": 0.5, "operator": "gt"}

        self.thresholder = Thresholder(**threshold_kwargs) if threshold_kwargs else None
        self.target_nan_mask = target_nan_mask
        self.cast_to_int = cast_to_int
        self.kwargs = kwargs

    def compute(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        target_nans = torch.isnan(target)

        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)

        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            target = target.clone()
            target[target_nans] = self.target_nan_mask
        elif self.target_nan_mask == "ignore-flatten":
            preds = preds[~target_nans]
            target = target[~target_nans]
        else:
            raise ValueError(f"Invalid target_nan_mask: {self.target_nan_mask}")

        if self.cast_to_int:
            target = target.int()

        return self.metric(preds, target, **self.kwargs)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(preds, target)

    def __repr__(self):
        if self.thresholder:
            return f"{self.metric_name}({self.thresholder})"
        return self.metric_name
