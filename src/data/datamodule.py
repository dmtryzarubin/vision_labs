import typing as ty
from functools import partial
from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import FaceDataset

__all__ = ["FaceDataModule"]


class FaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        data_folder: str,
        datadir_key: str,
        img_key: str,
        pts_key: str,
        box_key: str,
        normalize_keypoints: bool,
        additional_keys: ty.List[str],
        transforms: ty.Dict[str, A.Compose] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.target_key = target_key
        self.data_folder = Path(data_folder)
        self.img_key = img_key
        self.pts_key = pts_key
        self.box_key = box_key
        self.datadir_key = datadir_key
        self.additional_keys = additional_keys
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_keypoints = normalize_keypoints
        self._default_loader = partial(
            DataLoader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = FaceDataset(
                self.input_key,
                self.target_key,
                self.data_folder / "train.json",
                self.datadir_key,
                self.img_key,
                self.pts_key,
                self.box_key,
                self.additional_keys,
                self.transforms["train"],
                self.normalize_keypoints,
            )
            self.val_data = FaceDataset(
                self.input_key,
                self.target_key,
                self.data_folder / "val.json",
                self.datadir_key,
                self.img_key,
                self.pts_key,
                self.box_key,
                self.additional_keys,
                self.transforms["test"],
                self.normalize_keypoints,
            )
        if stage == "test" or stage is None:
            self.test_data = FaceDataset(
                self.input_key,
                self.target_key,
                self.data_folder / "test.json",
                self.datadir_key,
                self.img_key,
                self.pts_key,
                self.box_key,
                self.additional_keys,
                self.transforms["test"],
                self.normalize_keypoints,
            )

    def train_dataloader(self):
        return self._default_loader(dataset=self.train_data)

    def val_dataloader(self):
        return self._default_loader(dataset=self.val_data)

    def test_dataloader(self):
        return self._default_loader(dataset=self.test_data)
