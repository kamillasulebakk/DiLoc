import torch        # type: ignore
import numpy as np  # type: ignore
import tensorflow.keras.backend as K
import tensorflow as tf



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

def r2_score(y_true, y_pred):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()

    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    SS_res =  K.sum(K.square(y_true - y_pred)) # residual sum of squares
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) # total sum of squares
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
