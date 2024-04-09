import numpy as np

def prepare_data(data, history_window, pred_window, slide_step):
    x = []
    y = []

    for i in range(0, data.shape[1], slide_step):
        if i + history_window + pred_window > data.shape[1]:
            break
        x.append(data[:, i:i + history_window, :])
        y.append(data[:, i + history_window:i + history_window + pred_window, 0])

    x = np.array(x)
    y = np.array(y)

    return x, y