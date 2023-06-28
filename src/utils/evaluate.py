import os

import matplotlib.pyplot as plt
import pandas as pd
from beartype import beartype as typechecker
from beartype.typing import List

from .. import config
from ..preprocess.functional import open_json
from .visual import plot_ced


@typechecker
def evaluate(
    dlib_predictions_path: str,
    model_predictions_path: str,
    save_path_prefix: str = "",
    thresholds: List[float] = [0.08, 1.0],
) -> None:
    model_preds = pd.DataFrame(open_json(model_predictions_path))
    dlib_preds = pd.DataFrame(open_json(dlib_predictions_path))
    merged = pd.merge(model_preds, dlib_preds, how="inner", on=config.join_on)
    # include only for those images, where dlib detected face
    merged.dropna(inplace=True)
    for dataset_name, data in merged.groupby("dataset"):
        dlib_errors = data[config.dlib_nmse].values
        model_errors = data[config.model_nmse].values

        if dataset_name == "300W":
            kwargs = dict(errors=[model_errors], labels=[config.model_label])
        else:
            kwargs = dict(
                errors=[model_errors, dlib_errors],
                labels=[config.model_label, config.dlib_label],
            )
        for thr in thresholds:
            plot_ced(**kwargs, threshold=thr, dataset_name=dataset_name)
            save_path = os.path.join(save_path_prefix, "plots")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"CED_{thr:.2f}_{dataset_name}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
