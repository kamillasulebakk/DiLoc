import torch        # type: ignore
import numpy as np  # type: ignore
import load_data


def numpy_to_torch(a_f64):
    a_f32 = np.zeros(a_f64.shape, dtype=np.float32)
    a_f32[:, :] = a_f64
    tensor = torch.from_numpy(a_f32)
    return tensor

def normalize(x, max_x, min_x):
    x_new = (x - min_x)/(max_x - min_x)

    return x_new

def denormalize(x, max_x, min_x):
    x_new = x * (max_x - min_x) + min_x
    return x_new

def MSE(Y_true, Y_pred):
    error = np.square(np.subtract(Y_true,Y_pred)).mean()
    return error

# def relative_change(Y_true, Y_pred):
#     error = np.abs((Y_true - Y_pred)/(Y_true))*100
#     return error

def MAE(Y_true, Y_pred):
    error = np.abs(Y_pred - Y_true).mean()
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


def custom_loss(y_true, y_pred):
    eeg, target = load_data.load_data_files(10000, 'dipole_area', '1d', 1)

    x_max, y_max, z_max, r_max, a_max = target[:, 0].max(), target[:, 1].max(), target[:, 2].max(), target[:, 3].max(), target[:, 4].max()
    x_min, y_min, z_min, r_min, a_min = target[:, 0].min(), target[:, 1].min(), target[:, 2].min(), target[:, 3].min(), target[:, 4].min()

    # Extract the individual target variables
    x_true, y_true, z_true, radius_true, amplitude_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3], y_true[:, 4]
    x_pred, y_pred, z_pred, radius_pred, amplitude_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3], y_pred[:, 4]


    # Compute the loss for each variable using MSE for x, y, and z, and MAE for radius and amplitude
    x_loss = torch.nn.functional.mse_loss(x_pred, x_true)
    y_loss = torch.nn.functional.mse_loss(y_pred, y_true)
    z_loss = torch.nn.functional.mse_loss(z_pred, z_true)
    radius_loss = torch.nn.functional.mse_loss(radius_pred, radius_true)
    amplitude_loss = torch.nn.functional.mse_loss(amplitude_pred, amplitude_true)

    w_x = 1 / (x_max - x_min)
    w_y = 1 / (y_max - y_min)
    w_z = 1 / (z_max - z_min)
    w_r = 1 / (r_max - r_min)
    w_a = 1 / (a_max - a_min)

    # Compute the total loss as a weighted sum of the individual losses
    total_loss = w_x * x_loss + w_y * y_loss + w_z * z_loss + w_r * radius_loss + w_a * amplitude_loss

    return total_loss


def custom_loss_dipoles_w_amplitudes(Y_true, Y_pred, N_dipoles):

    eeg, target = load_data.load_data_files(10000, 'dipoles_w_amplitudes', '1d', N_dipoles)
    # target = target.T.reshape(10000, 4*N_dipoles)

    total_loss = 0

    for i in range(N_dipoles):

        x_max, y_max, z_max, a_max = target[:, 0 + (i*4)].max(), target[:, 1 + (i*4)].max(), target[:, 2 + (i*4)].max(), target[:, 3 + (i*4)].max()
        x_min, y_min, z_min, a_min = target[:, 0 + (i*4)].min(), target[:, 1 + (i*4)].min(), target[:, 2 + (i*4)].min(), target[:, 3 + (i*4)].min()

        # Extract the individual target variables
        x_true, y_true, z_true, amplitude_true = Y_true[:, 0 + (i*4)], Y_true[:, 1 + (i*4)], Y_true[:, 2 + (i*4)], Y_true[:, 3 + (i*4)]
        x_pred, y_pred, z_pred, amplitude_pred = Y_pred[:, 0 + (i*4)], Y_pred[:, 1 + (i*4)], Y_pred[:, 2 + (i*4)], Y_pred[:, 3 + (i*4)]

        x_loss = torch.nn.functional.mse_loss(x_pred, x_true)
        y_loss = torch.nn.functional.mse_loss(y_pred, y_true)
        z_loss = torch.nn.functional.mse_loss(z_pred, z_true)
        amplitude_loss = torch.nn.functional.mse_loss(amplitude_pred, amplitude_true)

        w_x = 1 / (x_max - x_min)
        w_y = 1 / (y_max - y_min)
        w_z = 1 / (z_max - z_min)
        w_a = 1 / (a_max - a_min)

        # Compute the total loss as a weighted sum of the individual losses
        loss = w_x * x_loss + w_y * y_loss + w_z * z_loss + w_a * amplitude_loss

        total_loss += 1/N_dipoles * loss

    return total_loss