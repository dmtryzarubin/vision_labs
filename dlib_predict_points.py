import os
from pathlib import Path
from typing import Any, Dict, List

import dlib
import fire
import numpy as np
from tqdm.autonotebook import tqdm

from src import config
from src.metrics.functional import calc_norm_coef, normalized_mse
from src.preprocess.box_utils import get_height, get_width
from src.preprocess.dlib_detection import DlibPointsDetector
from src.preprocess.functional import open_json, save_json


def _predict_points(
    data: List[Dict[str, Any]], predictor: DlibPointsDetector, verbose: bool = True
) -> List[Dict[str, Any]]:
    pbar = tqdm(data, disable=not verbose)
    predictions = [{}] * len(data)
    for i, sample in enumerate(pbar):
        ans = {}
        if sample["dlib_face_detected"]:
            img_path = os.path.join(sample["datadir"], sample["img_path"])
            image = dlib.load_rgb_image(img_path)
            pts = predictor(image, sample["dlib_face_box"])
            box = sample["dlib_face_box"]
            h, w = get_height(box), get_width(box)
            d = calc_norm_coef(h, w)
            targets = np.array(sample["pts"])
            pts = np.array(pts)
            ans["dlib_nmse"] = normalized_mse(pts, targets, d)
            ans["dlib_pts"] = pts.tolist()
        else:
            ans["dlib_nmse"] = float("nan")
            ans["dlib_pts"] = []
        prediction = {
            "dataset": sample["dataset"],
            "split": sample["split"],
            "fname": sample["fname"],
        }
        prediction.update(ans)
        predictions[i] = prediction
    return predictions


def predict_points(
    dataset_path: str, predictor_path: str = config.PREDICTOR_PATH, verbose: bool = True
) -> None:
    data = open_json(dataset_path)
    predictor = DlibPointsDetector(predictor_path)
    predictions = _predict_points(data, predictor, verbose)
    datadir = Path(dataset_path).parent
    save_json(predictions, datadir / "dlib_test_predictions.json")


if __name__ == "__main__":
    fire.Fire(predict_points)
