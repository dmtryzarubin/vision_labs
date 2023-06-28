from pathlib import Path
from typing import Union

import fire
from sklearn.model_selection import train_test_split

from src.preprocess.functional import open_json, save_json

RANDOM_STATE = 2023


def split(dataset_path: str, val_size: Union[float, int] = 500) -> None:
    datadir = Path(dataset_path).parent
    data = open_json(dataset_path)
    test = [d for d in data if d["split"] == "test"]
    train = [d for d in data if d["split"] == "train"]
    train, val = train_test_split(train, test_size=val_size, random_state=RANDOM_STATE)
    paths = {
        "train": datadir / "train.json",
        "val": datadir / "val.json",
        "test": datadir / "test.json",
    }
    save_json(train, paths["train"])
    save_json(val, paths["val"])
    save_json(test, paths["test"])
    return paths


if __name__ == "__main__":
    fire.Fire(split)
