
import numpy as np
import torch
def MAE(data, ground_truth):
    temp = data - ground_truth
    number = data.numel()
    abs_data = np.abs(temp.detach().numpy())
    mae = abs_data.sum() / number
    return mae


def RMSE(data, ground_truth):
    temp = data - ground_truth
    temp_2 = pow(temp, 2)
    number = data.numel()
    rmse = pow(temp_2.sum() / number, 1 / 2)
    return rmse


def MAPE(data, ground_truth):
    temp = data - ground_truth
    new_data = temp / ground_truth
    abs_data = np.abs(new_data.detach().numpy())
    sum_data = 0
    number = ground_truth.numel()
    for i in range(ground_truth.shape[0]):
        for j in range(ground_truth.shape[1]):
            for k in range(ground_truth.shape[2]):
                if (ground_truth[i, j, k] == 0):
                    number -= 1
                    continue;
                sum_data += abs_data[i, j, k]
    mape = sum_data / number
    return mape


def CCC(data, ground_truth):
    data = data.reshape([-1])
    ground_truth = ground_truth.reshape([-1])
    cor = np.corrcoef(data.detach().numpy(), ground_truth)[0, 1]
    data_mean = torch.mean(data)
    ground_truth_mean = torch.mean(ground_truth)
    data_std = torch.std(data)
    ground_truth_std = torch.std(ground_truth)
    data_var = torch.var(data)
    ground_truth_var = torch.var(ground_truth)
    top = 2 * cor * data_std * ground_truth_std
    bottom = data_var + ground_truth_var + (data_mean - ground_truth_mean) ** 2
    ccc = top / bottom
    return ccc


def PCC(data, ground_truth):
    data_mean = torch.mean(data)
    ground_truth_mean = torch.mean(ground_truth)
    top = ((data - data_mean) * (ground_truth - ground_truth_mean)).sum()
    bottom = pow(((ground_truth - ground_truth_mean) ** 2).sum(), 1 / 2) * pow(((data - data_mean) ** 2).sum(), 1 / 2)
    pcc = top / bottom
    return pcc