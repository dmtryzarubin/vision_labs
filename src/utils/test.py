import numpy as np
import pytorch_lightning as pl
import torch
from beartype import beartype as typechecker
from beartype.typing import Any, Dict, List
from tqdm.autonotebook import tqdm

from .. import config
from ..data.dataset import FaceDataset
from ..metrics.functional import calc_norm_coef, normalized_mse
from ..model.pl_model import FaceModel
from ..preprocess.box_utils import get_height, get_width
from .postprocess import postprocess


@typechecker
@torch.no_grad()
def predict(
    model: FaceModel, dataset: FaceDataset, device: str, verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Inferences model on a dataset and creates List of dicts with prediction

    :param model: FaceModel
    :param dataset: FaceDataset
    :param device: device to infer model on
    :param verbose: Verbose param, defaults to True
    :return: List of dict with predicted points and nmse
    """
    model.to(device)
    model.eval()
    preds = []
    pbar = tqdm(range(len(dataset)), disable=not verbose)
    for i in pbar:
        sample = dataset[i]
        output = model.predict(
            sample["image"][
                None,
            ].to(device)
        ).cpu()
        output = output[0].numpy()
        box = sample[dataset.box_key]
        h, w = get_height(box), get_width(box)
        d = calc_norm_coef(h, w)
        output = postprocess(output, box, h, w, sample["image"].shape[1:])
        pred_dict = {k: sample[k] for k in config.KEYSTOKEEP}
        pred_dict[config.model_nmse] = normalized_mse(
            output, np.array(sample["pts"]), d
        )
        pred_dict[config.model_pts] = output.tolist()
        preds.append(pred_dict)
    return preds
