import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import albumentations as A
import dlib
import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocess.box_utils import crop_by_box
from ..preprocess.functional import open_json


class FaceDataset(Dataset):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        dataset_path: str,
        datadir_key: str,
        img_key: str,
        pts_key: str,
        box_key: str,
        additional_keys: List[str],
        transforms: A.Compose,
        normalize_keypoints: bool = True,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.target_key = target_key
        self.data = open_json(dataset_path)
        self.data_dir = Path(dataset_path).parent
        self.transforms = transforms
        self.img_key = img_key
        self.pts_key = pts_key
        self.box_key = box_key
        self.datadir_key = datadir_key
        self.normalize_keypoints = normalize_keypoints
        self.additional_keys = additional_keys

    def __len__(self) -> int:
        return len(self.data)

    def _preload_data(self, idx: int) -> tuple:
        sample = deepcopy(self.data[idx])
        img_path = os.path.join(self.data_dir, sample[self.img_key])
        img = dlib.load_rgb_image(img_path)
        img = crop_by_box(img, sample[self.box_key])
        sample[self.input_key] = img
        sample[self.pts_key] = np.array(sample[self.pts_key])
        sub = np.array([sample[self.box_key][0], sample[self.box_key][1]])
        sample[self.target_key] = sample[self.pts_key] - sub
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._preload_data(idx)
        transformed = self.transforms(
            image=sample[self.input_key], keypoints=sample[self.target_key]
        )
        transformed[self.target_key] = torch.tensor(
            transformed[self.target_key], dtype=torch.float
        ).view(-1)
        transformed[self.pts_key] = sample[self.pts_key]
        transformed[self.box_key] = sample[self.box_key]
        for key in self.additional_keys:
            transformed[key] = sample[key]
        if self.normalize_keypoints:
            # TODO change scaling per dim for non-square images
            transformed[self.target_key] /= transformed[self.input_key].shape[1]
        return transformed
