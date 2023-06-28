import json
import os
from glob import glob
from pathlib import Path

import dlib
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Any, Dict, Iterable, List, Tuple, Union
from tqdm.autonotebook import tqdm

from . import box_utils
from .dlib_detection import DlibFaceDetector


@typechecker
def open_json(path: Union[Path, str]) -> Iterable:
    with open(path, "r") as f:
        data = json.load(f)
    return data


@typechecker
def save_json(
    data: List[Dict[str, Any]], path: Union[Path, str], overwrite: bool = True
) -> None:
    if os.path.exists(path) and not overwrite:
        raise ValueError
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


@typechecker
def readlines(path: Union[Path, str]):
    with open(path, "r") as f:
        lines = f.readlines()
    return lines


@typechecker
def get_n_points(path: Union[Path, str]) -> int:
    lines = readlines(path)
    return int(lines[1].split()[1])


@typechecker
def correct_img_fname(path: Path, fname: str) -> str:
    if os.path.exists(path / f"{fname}.jpg"):
        img_name = f"{fname}.jpg"
    elif os.path.exists(path / f"{fname}.jpeg"):
        img_name = f"{fname}.jpeg"
    else:
        img_name = f"{fname}.png"
    return img_name


@typechecker
def get_pts(path: Path) -> List[List[float]]:
    lines = readlines(path)
    pts = []
    for line in lines[3:-1]:
        pts.append([float(x) for x in line.split()])
    return pts


@typechecker
def process_box_from_pts(
    pts: List[List[float]], shape: Tuple[int, int]
) -> List[Union[int, float]]:
    box = box_utils.box_from_pts(pts)
    box = box_utils.round_box(box)
    box = box_utils.check_box(box, shape)
    return box


@typechecker
def process_single_path(path: str, datadir: str) -> Dict[str, Any]:
    path = Path(path)
    parts = path.parts
    parent = path.parent
    dataset, split, pts_fname = parts[-3:]
    fname = pts_fname.replace(".pts", "")
    img_fname = correct_img_fname(parent, fname)
    shape = dlib.load_rgb_image(os.path.join(parent, img_fname)).shape
    shape = shape[:2]
    pts = get_pts(path)
    box = process_box_from_pts(pts, shape)
    img_path = os.path.join(dataset, split, img_fname)
    pts_path = os.path.join(dataset, split, pts_fname)
    sample = {
        "datadir": datadir,
        "dataset": dataset,
        "split": split,
        "fname": fname,
        "img_path": img_path,
        "pts_path": pts_path,
        "face_box": box,
        "n_pts": len(pts),
        "pts": pts,
    }
    return sample


@typechecker
def read_image(datdir: str, inner_path: str) -> np.ndarray:
    path = os.path.join(datdir, inner_path)
    return dlib.load_rgb_image(path)


@typechecker
def create_annotation(
    datadir: str,
    pattern: str = "*/*/*.pts",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    disable = not verbose
    paths = glob(os.path.join(datadir, pattern))
    data = []
    pbar = tqdm(paths, disable=disable)
    face_det = DlibFaceDetector()
    for i, path in enumerate(pbar):
        sample = process_single_path(path, datadir)
        if sample["n_pts"] != 68:
            continue
        image = read_image(sample["datadir"], sample["img_path"])
        face_det_result = face_det(image, sample["face_box"])
        sample.update(face_det_result)
        data.append(sample)
    return data
