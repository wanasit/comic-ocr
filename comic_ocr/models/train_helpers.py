from pathlib import Path
from typing import Dict, Callable, List

import torch

"""A dictionary of metrics, where each metric is the list of values from the beginning to latest update."""
HistoricalMetrics = Dict[str, List[float]]

"""A callback function that is called on reaching update step count."""
UpdateCallback = Callable[[int, HistoricalMetrics, HistoricalMetrics], None]


def callback_to_save_model(model, model_path) -> UpdateCallback:
    def save(steps, training_metrics, validation_metrics):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.to('cpu')
        torch.save(model, model_path)

    return save


def callback_to_save_model_on_increasing_metric(model, model_path, metric_name) -> UpdateCallback:
    save_model_callback = callback_to_save_model(model, model_path)
    metric_value = []

    def save(steps, training_metrics, validation_metrics):
        if not metric_value or metric_value[0] < validation_metrics[metric_name][-1]:
            save_model_callback(steps, training_metrics, validation_metrics)
            metric_value.clear()
            metric_value.append(validation_metrics[metric_name][-1])

    return save
