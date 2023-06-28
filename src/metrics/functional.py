import numpy as np
from beartype import beartype as typechecker
from beartype.typing import List, Tuple, Union
from jaxtyping import Float, Int


@typechecker
def calc_norm_coef(box_h: Union[int, float], box_w: Union[int, float]) -> float:
    return (box_h * box_w) ** 0.5


@typechecker
def normalized_mse(
    output: Union[Float[np.ndarray, "68 2"], Int[np.ndarray, "68 2"]],
    target: Union[Float[np.ndarray, "68 2"], Int[np.ndarray, "68 2"]],
    d: float,
) -> float:
    """
    Calculates normalized rmse between outputs and targets

    :param output: Face landmarks predicted by model 
    :param target: Target face landmarks
    :param d: Normalization factor
    :return: NMSE value for sample 
    """
    # 68x2 -> 68x2
    mse = (output - target) ** 2
    # 68x2 -> 68
    mse = np.sqrt(mse.sum(axis=1))
    # 68 -> 1
    return mse.sum() / (len(mse) * d)


@typechecker
def calc_proportions(
    errors: Float[np.ndarray, "batch"]
) -> Tuple[Float[np.ndarray, "batch"], Float[np.ndarray, "batch"]]:
    """
    Calculates x and y values for empirical cumulative distribution plot

    :return: Tuple with x and y values
    """
    errors = np.sort(errors)
    n = len(errors)
    values = np.zeros_like(errors)
    for i, thr in enumerate(errors):
        mask = errors < thr
        numel = np.count_nonzero(mask)
        pct = numel / n
        values[i] = pct
    return errors, values


@typechecker
def calc_auc(
    errors: Float[np.ndarray, "batch"], threshold: float = 0.08, step: float = 0.001
) -> float:
    """
    Calculates auc for given sorted array of errors

    :param errors: Array with errors on dataset
    :param threshold: Threshold, too calc error up to, defaults to 0.08
    :param step: Step value for integration, defaults to 0.001
    :return: Float with AUC
    """
    n = len(errors)
    auc = 0.0
    for thr in np.arange(0.0, threshold, step):
        counts = np.count_nonzero(errors < thr)
        pct = counts / n
        area = pct * step
        auc += area
    return auc
