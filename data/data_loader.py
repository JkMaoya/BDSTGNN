import torch
import numpy as np
from utils.tools import *
def data_processing(loc_list,raw_data,test_window,valid_window,history_window,pred_window,slide_step,device):
    active_cases = []
    confirmed_cases = []



    static_feat = []
    for i, each_loc in enumerate(loc_list):
        active_cases.append(raw_data[raw_data['state'] == each_loc]['active'])
        confirmed_cases.append(raw_data[raw_data['state'] == each_loc]['confirmed'])
        static_feat.append(np.array(raw_data[raw_data['state'] == each_loc][['population', 'density', 'lng', 'lat']]))
    active_cases = np.array(active_cases)
    confirmed_cases = np.array(confirmed_cases)

    static_feat = np.array(static_feat)[:, 0, :]
    recovered_cases = confirmed_cases - active_cases
    susceptible_cases = np.expand_dims(static_feat[:, 0], -1) - active_cases - recovered_cases

    normalizer = {'S': {}, 'I': {}, 'R': {}}

    for i, each_loc in enumerate(loc_list):
        normalizer['S'][each_loc] = (np.max(susceptible_cases[i]), np.min(susceptible_cases[i]))
        normalizer['I'][each_loc] = (np.max(active_cases[i]), np.min(active_cases[i]))
        normalizer['R'][each_loc] = (np.max(recovered_cases[i]), np.min(recovered_cases[i]) + 10)


    data_truth=np.concatenate((np.expand_dims(active_cases, axis=-1), np.expand_dims(recovered_cases, axis=-1),
                               np.expand_dims(susceptible_cases, axis=-1)), axis=-1)
    data_feat=data_truth.copy()


    for i, each_loc in enumerate(loc_list):
        data_feat[i, :, 0] = (data_feat[i, :, 0] - normalizer['I'][each_loc][1]) / (normalizer['I'][each_loc][0]-normalizer['I'][each_loc][1])
        data_feat[i, :, 1] = (data_feat[i, :, 1] - normalizer['R'][each_loc][1]) / (normalizer['R'][each_loc][0]-normalizer['R'][each_loc][1])
        data_feat[i, :, 2] = (data_feat[i, :, 2] - normalizer['S'][each_loc][1]) / (normalizer['S'][each_loc][0]-normalizer['S'][each_loc][1])
    I_max = []
    I_min = []
    R_max = []
    R_min = []
    S_max =[]
    S_min = []


    for i, each_loc in enumerate(loc_list):
        I_max.append(normalizer['I'][each_loc][0])
        R_max.append(normalizer['R'][each_loc][0])
        S_max.append(normalizer['S'][each_loc][0])
        I_min.append(normalizer['I'][each_loc][1])
        R_min.append(normalizer['R'][each_loc][1])
        S_min.append(normalizer['S'][each_loc][1])

    I_max = np.array(I_max)
    I_min = np.array(I_min)
    R_max = np.array(R_max)
    R_min = np.array(R_min)
    S_max = np.array(S_max)
    S_min = np.array(S_min)

    train_feat = data_feat[:, :-valid_window-test_window, :]
    val_feat = data_feat[:, -valid_window-test_window:-test_window, :]
    test_feat = data_feat[:, -test_window:, :]

    train_truth=data_truth[:, :-valid_window-test_window, :]
    val_truth=data_truth[:, -valid_window-test_window:-test_window, :]
    test_truth=data_truth[:, -test_window:, :]

    train_x,train_y=prepare_data(train_feat, history_window, pred_window, slide_step)
    train_x_true,_=prepare_data(train_truth, history_window, pred_window, slide_step)
    test_x,_=prepare_data(test_feat, history_window, pred_window, slide_step)
    _,test_y=prepare_data(test_truth, history_window, pred_window, slide_step)
    test_x_true,_=prepare_data(test_truth, history_window, pred_window, slide_step)
    val_x,val_y=prepare_data(val_feat, history_window, pred_window, slide_step)
    val_x_true,_=prepare_data(val_truth, history_window, pred_window, slide_step)
    train_x = torch.tensor(train_x).to(device).to(torch.float32)
    train_y = torch.tensor(train_y).to(device).to(torch.float32)

    train_x_true = torch.tensor(train_x_true).to(device).to(torch.float32)

    val_x = torch.tensor(val_x).to(device).to(torch.float32)
    val_y = torch.tensor(val_y).to(device).to(torch.float32)
    val_x_true = torch.tensor(val_x_true).to(device).to(torch.float32)
    test_x = torch.tensor(test_x).to(device).to(torch.float32)
    test_y = torch.tensor(test_y).to(device).to(torch.float32)
    test_x_true = torch.tensor(test_x_true).to(device).to(torch.float32)
    I_max = torch.tensor(I_max, dtype=torch.float32).to(device).reshape((I_max.shape[0])).to(device)
    I_min = torch.tensor(I_min, dtype=torch.float32).to(device).reshape((I_min.shape[0])).to(device)
    R_max = torch.tensor(R_max, dtype=torch.float32).to(device).reshape((R_max.shape[0])).to(device)
    R_min = torch.tensor(R_min, dtype=torch.float32).to(device).reshape((R_min.shape[0])).to(device)
    S_max = torch.tensor(S_max, dtype=torch.float32).to(device).reshape((S_max.shape[0])).to(device)
    S_min = torch.tensor(S_min, dtype=torch.float32).to(device).reshape((S_min.shape[0])).to(device)

    N = torch.tensor(static_feat[:, 0], dtype=torch.float32).to(device)

    return train_x,train_y,train_x_true,val_x,val_y,val_x_true,test_x,test_y,test_x_true,S_min, S_max, I_min, I_max, R_min,R_max, N