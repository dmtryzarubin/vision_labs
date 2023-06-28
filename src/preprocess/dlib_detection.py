import dlib
import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Any, Dict, Iterable, List, Tuple, Union

from .. import config
from .box_utils import box_to_rect, correct_boxes, get_boxes


@typechecker
def detect_face(
    detector: Any, img: np.ndarray, face_box: List[Union[int, float]]
) -> Tuple[Iterable[Iterable[Union[float, int]]], bool]:
    face_detected = True
    dets = detector(img, 1)
    if not len(dets):
        face_detected = False
    boxes = get_boxes(dets, img.shape[:2])
    boxes = correct_boxes(boxes, face_box, img.shape[:2])
    return boxes, face_detected


def predict_face_points(predictor: Any, img: np.ndarray, face_box: List[int]):
    assert len(face_box) == 4
    det = box_to_rect(face_box)
    points = predictor(img, det).parts()
    shape = [[point.x, point.y] for point in points]
    return shape


class DlibFaceDetector:
    keepfirst = True
    box_key = "dlib_face_box"
    face_det_key = "dlib_face_detected"

    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, image: np.ndarray, face_box: List[int]) -> Dict[str, Any]:
        boxes, face_detected = detect_face(self.detector, image, face_box)
        if self.keepfirst:
            boxes = boxes[0]
        return {self.box_key: boxes, self.face_det_key: face_detected}


class DlibPointsDetector:
    def __init__(self, predictor_path: str = config.PREDICTOR_PATH) -> None:
        self.predictor = dlib.shape_predictor(predictor_path)

    def __call__(
        self, image: np.ndarray, face_box: List[int]
    ) -> Dict[str, List[List[int]]]:
        points = predict_face_points(self.predictor, image, face_box)
        return points
