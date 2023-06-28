import os

import fire

import src.preprocess.functional as F


def create_annotation(
    datadir: str, pattern: str = "*/*/*.pts", verbose: bool = True
) -> str:
    """
    Creates annotation for ./landmarks_task datadir

    :param datadir: Path to ./landmarks_task datadir
    :param pattern: Pattern to filter `.pts` files, defaults to "*/*/*.pts"
    :param verbose: Boolean flag for printing progress bar, defaults to True
    :return: Path, where dataset was saved
    """
    data = F.create_annotation(datadir, pattern, verbose)
    save_path = os.path.join(datadir, "dataset.json")
    F.save_json(data, save_path)
    return save_path


if __name__ == "__main__":
    fire.Fire(create_annotation)
