import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Any, Dict, List
from jaxtyping import Float

from ..metrics.functional import calc_auc, calc_proportions

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 150


@typechecker
def plot_ced(
    errors: List[Float[np.ndarray, "batch"]],
    labels: List[str],
    dataset_name: str = "",
    threshold: float = 0.08,
    step: float = 0.01,
    fig_kwargs: Dict[str, Any] = {},
) -> None:
    """
    PLots Cumulative Error Distribution for a given list of errors

    :param errors: List of arrays with errors
    :param labels: List of labels for each array in errors
    :param dataset_name: Dataset name, defaults to ""
    :param threshold: Threshold to calculate AUC on, defaults to 0.08
    :param step: Step of integration for AUC, defaults to 0.01
    :param fig_kwargs: Matplotlib figure kwargs, defaults to {}
    """
    assert len(labels) == len(errors)
    plt.figure(**fig_kwargs)
    s = ""
    max_x = 0.0
    for label, error in zip(labels, errors):
        x, y = calc_proportions(error)
        auc = calc_auc(x, threshold, step)
        s += f"AUC at {threshold}: {auc:.3f} for {label}\n"
        plt.plot(x, y, label=label)
        # plt.fill_between(x, y, step="pre", alpha=0.2)
        max_x = max(x.max(), max_x)
    plt.text(
        0.04,
        0.1,
        s,
        size=10,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round",
            ec=(1.0, 0.5, 0.5),
            fc=(1.0, 0.8, 0.8),
        ),
    )
    plt.title(f"Cumulative Error Distribution {dataset_name}")
    plt.xlabel("NME")
    plt.ylabel("Fraction of a dataset")
    plt.legend()
    # plt.xlim([0.0, min(threshold, max_x)])
    plt.xlim([0.0, 0.08])
    plt.ylim([0.0, 1.0])
