import torch        # type: ignore
import numpy as np  # type: ignore


def numpy_to_torch(a_f64):
    a_f32 = np.zeros(a_f64.shape, dtype=np.float32)
    a_f32[:, :] = a_f64
    tensor = torch.from_numpy(a_f32)
    return tensor


def indices_from_positions(positions: np.ndarray, max_index: int) -> np.ndarray:
    pos = positions.copy()
    pos -= pos.min()
    pos *= max_index/pos.max()
    indices = pos.astype(np.intc)
    return indices
