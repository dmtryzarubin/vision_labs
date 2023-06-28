import os
from typing import Union

import fire

from src import config
from src.preprocess.functional import create_annotation

from create_annotation import create_annotation
from dlib_predict_points import predict_points
from train_test_split import split


def main(
    datadir: str,
    pattern: str = "*/*/*.pts",
    val_size: Union[float, int] = 500,
    predictor_path: str = config.PREDICTOR_PATH,
    verbose: bool = True,
) -> None:
    print("Creating annotation & predicting face_box by dlib...")
    dataset_path = create_annotation(datadir, pattern, verbose)
    print("Splitting dataset into train/val/test...")
    paths = split(dataset_path, val_size)
    print("Predicting face points by dlib...")
    predict_points(paths["test"], predictor_path, verbose)


if __name__ == "__main__":
    fire.Fire(main)
