from math import ceil

import dlib
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Any, Iterable, List, Tuple, Union


@typechecker
def box_from_pts(face_pts: List[List[float]]) -> List[float]:
    """
    Creates bounding box from face landmark points

    :param face_pts: List of lists with x, y coordinates
    :return: List with [xmin, ymin, xmax, ymax] box coords
    """
    face_pts = np.array(face_pts)
    bbox = [
        face_pts[:, 0].min(),
        face_pts[:, 1].min(),
        face_pts[:, 0].max(),
        face_pts[:, 1].max(),
    ]
    bbox = [float(x) for x in bbox]
    return bbox


@typechecker
def round_box(box: List[Union[float, int]]) -> List[int]:
    """
    Round bounding box coordinates

    :param box: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    :return: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    """
    return [int(box[0]), int(box[1]), ceil(box[2]), ceil(box[3])]


@typechecker
def crop_by_box(img: np.ndarray, box: List[int]) -> np.ndarray:
    """
    Crop box from image

    :param img: Image to crop box from
    :param box: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    :return: Cropped region from image
    """
    return img[box[1] : box[3], box[0] : box[2]]


@typechecker
def check_box(box: Iterable, shape: tuple) -> Iterable:
    """
    Checks if bounding bos is out of bounds

    :param box: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    :param shape: Tuple with image (H, W) shape
    :return: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    """
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], shape[1])
    box[3] = min(box[3], shape[0])
    return box


@typechecker
def get_boxes(dets: Any, shape: Tuple[int, int]) -> List[List[Union[float, int]]]:
    """
    Transforms dlib.rectangle to bounding box cordinates in [xmin, ymin, xmax, ymax] format

    :param dets: List of dlib.rectaingles
    :param shape: Tuple with image (H, W) shape
    :return: List of bounding box cordinates in [xmin, ymin, xmax, ymax]
    """
    bboxes = []
    for d in dets:
        # xmin ymin, xmax, ymax
        box = [
            max(d.left(), 0),
            max(d.top(), 0),
            min(d.right(), shape[1]),
            min(d.bottom(), shape[0]),
        ]
        bboxes.append(box)
    return bboxes


@typechecker
def pad_box(box: Iterable, pad: int = 10) -> Iterable:
    """
    Pads bounding box by given `pad` value

    :param box: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    :param pad: Pad value, defaults to 10
    :return: Bounding box cordinates in [xmin, ymin, xmax, ymax] format
    """
    box[0] -= pad
    box[1] -= pad
    box[2] += pad
    box[3] += pad
    return box


@typechecker
def correct_boxes(
    boxes: Iterable[Iterable[Union[float, int]]],
    face_box: List[int],
    shape: Tuple[int, int],
    pad: int = 10,
) -> Iterable[Iterable[Union[float, int]]]:
    """
    Function that corrects bounding box from dlib

    :param boxes: List of predicted boxes
    :param face_box: Face bbox obtained from landmark points
    :param shape: Image shape
    :param pad: Pad value, defaults to 10
    :return: List of checled bounding boxes
    """
    # if dlib hasn't find any boxes, then use face points
    if not len(boxes):
        face_box = pad_box(face_box, pad)
        face_box = check_box(face_box, shape)
        return [face_box]

    for i, box in enumerate(boxes):
        box[0] = min(box[0], face_box[0])
        box[1] = min(box[1], face_box[1])
        box[2] = max(box[2], face_box[2])
        box[3] = max(box[3], face_box[3])
        boxes[i] = check_box(box, shape)
    return boxes


@typechecker
def get_height(box: List[Union[float, int]]) -> Union[float, int]:
    """
    Calculates height from bounding box in [xmin, ymin, xmax, ymax] format

    :param box: Bounding box in [xmin, ymin, xmax, ymax] format
    :return: height  in pixels
    """
    return box[3] - box[1]


@typechecker
def get_width(box: List[Union[float, int]]) -> Union[float, int]:
    """
    Calculates width from bounding box in [xmin, ymin, xmax, ymax] format

    :param box: Bounding box in [xmin, ymin, xmax, ymax] format
    :return: Width  in pixels
    """
    return box[2] - box[0]


@typechecker
def calc_scale(box_size: Union[int, float], image_size: int) -> Union[int, float]:
    """
    Calculates scale after resizing
    """
    return box_size / image_size


@typechecker
def box_to_rect(box: List[Union[float, int]]) -> dlib.rectangle:
    """
    Coverts bounding box into dlib.rectaingle

    :param box: Bounding box in [xmin, ymin, xmax, ymax] format
    :return: dlib.rectaingle
    """
    return dlib.rectangle(*box)
