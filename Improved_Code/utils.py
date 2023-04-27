import torch        # type: ignore
import numpy as np  # type: ignore
# import tensorflow.keras.backend as K
# import tensorflow as tf



def numpy_to_torch(a_f64):
    a_f32 = np.zeros(a_f64.shape, dtype=np.float32)
    a_f32[:, :] = a_f64
    tensor = torch.from_numpy(a_f32)
    return tensor

def normalize(x):
    x_new = (x - np.min(x))/(np.max(x) - np.min(x))
    return x_new

# def denormalize(x, max_x, min_x):
#     x_new = x * (max_x - min_x) + min_x
#     return x_new

def denormalize(x, max_x, min_x):
    x_new = x * (max_x - min_x) + min_x
    return x_new

def MSE(Y_true, Y_pred):
    error = np.square(np.subtract(Y_true,Y_pred)).mean()
    return error

def relative_change(Y_true, Y_pred):
    error = (Y_pred - Y_true)/np.abs(Y_true)
    return error

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


def xz_plane_idxs(nyhead, xz_plane_idxs):
    eeg_list = []
    x_pred_list = []
    y_pred_list = []
    z_pred_list = []

    for i, idx in enumerate(xz_plane_idxs):
        nyhead.set_dipole_pos(nyhead.cortex[:,idx])
        eeg = calculate_eeg(nyhead)
        eeg = (eeg - np.mean(eeg))/np.std(eeg)
        eeg = numpy_to_torch(eeg.T)
        eeg_list.append(eeg)
        pred = model(eeg)
        pred = pred.detach().numpy().flatten()

        x_max, x_min = find_max_min(0)
        y_max, y_min = find_max_min(1)
        z_max, z_min = find_max_min(2)

        # denormalize target coordinates
        x_pred_list[i, 0] = denormalize(pred[0], x_max, x_min)
        y_pred_list[i, 1] = denormalize(pred[1], y_max, y_min)
        z_pred_list[i, 2] = denormalize(pred[2], z_max, z_min)

        x_target_list[i] = nyhead.cortex[0,idx]
        y_target_list[i] = nyhead.cortex[1,idx]
        z_target_list[i] = nyhead.cortex[2,idx]

    return eeg_list, x_pred_list, y_pred_list, z_pred_list