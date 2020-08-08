import numpy as np


def quat_fix(q):
    """Enforce quaternion continuity across the time dimension.

    Borrowed from QuaterNet:
    https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L119

    This function falls under the Attribution-NonCommercial 4.0 International
    license.

    Selects the representation (q or -q) with minimal distance
    (or, equivalently, maximal dot product) between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and
    J is the number of joints.
    Returns a tensor of the same shape.

    Args:
        q (np.ndarray): quaternions of size (L, J, 4) to enforce continuity.

    Returns:
        np.ndarray: quaternion of size (L, J, 4) that is continuous
            in time dimension.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result
