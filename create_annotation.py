import os

import fire

import src.preprocess.functional as F


def create_annotation(
    datadir: str, pattern: str = "*/*/*.pts", verbose: bool = True
) -> str:
    data = F.create_annotation(datadir, pattern, verbose)
    save_path = os.path.join(datadir, "dataset.json")
    F.save_json(data, save_path)
    return save_path


if __name__ == "__main__":
    fire.Fire(create_annotation)
