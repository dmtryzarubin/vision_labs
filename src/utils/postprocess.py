import numpy as np
from beartype import beartype as typechecker
from beartype.typing import Iterable, List, Union
from jaxtyping import Float

from ..preprocess.box_utils import calc_scale


@typechecker
def postprocess(
    output: Float[np.ndarray, "68 2"],
    box: List[int],
    box_h: Union[int, float],
    box_w: Union[int, float],
    image_shape: Iterable[int],
) -> np.ndarray:
    h_scale = calc_scale(box_h, image_shape[0])
    w_scale = calc_scale(box_w, image_shape[1])
    output[:, 0] *= w_scale
    output[:, 1] *= h_scale
    output[:, 0] += box[0]
    output[:, 1] += box[1]
    return output
