
import numpy as np
import pandas as pd
from torch.nn.utils import weight_norm
from model.BDSTGN import *
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from utils.metrics import *
from utils.tools import *
from data.data_loader import *


from tqdm import tqdm

seed = 1234
pred_window=5
history_window=5
slide_step=5
test_window = 50
valid_window = 50

torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

import torch.optim as optim



raw_data = pickle.load(open('data/state_covid_data.pickle','rb'))
pop_data = pd.read_csv('data/uszips.csv')
pop_data = pop_data.groupby('state_name').agg({'population':'sum', 'density':'mean', 'lat':'mean', 'lng':'mean'}).reset_index()
raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

loc_list = list(raw_data['state'].unique())

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

(train_x,train_y,train_x_true,val_x,val_y,val_x_true,test_x,test_y,test_x_true,
 S_min, S_max, I_min, I_max, R_min,R_max, N)=(
    data_processing(loc_list,raw_data,test_window,valid_window,history_window,pred_window,slide_step,device))





num_nodes=train_x.shape[1]
node_embed=32
kernel_size=15
out_dim_gcn=32
num_channels_graph=[8]
kernel_size_graph=3
dropout=0



min_loss = 1e10
model = BDSTGN(node_embed, kernel_size, out_dim_gcn, history_window, pred_window, num_nodes, num_channels_graph,
               kernel_size_graph, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

criterion = nn.L1Loss(reduction='mean')

T = 30
patience_count = 0
patience = 10
min_delta = 0.01

for epoch in tqdm(range(200)):
    for batch in range(train_x.shape[0]):
        model.train()

        pred_nn_I_res_train, pred_phy_I_res_train = model(train_x[batch], train_x_true[ batch],S_min, S_max,I_min, I_max,
                                                                                                      R_min, R_max, N,device)

        pred_nn_I_res_train = pred_nn_I_res_train.squeeze()
        ww = [1, 1]
        loss = ww[0] * criterion(pred_nn_I_res_train, train_y[batch]) + criterion(pred_phy_I_res_train, train_y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

    val_pred = []
    val_phy = []
    for batch_val in range(val_x.shape[0]):
        pred_nn_I_res_val, pred_phy_I_res_val = model(val_x[batch_val], val_x_true[batch_val],
                                                                          S_min, S_max, I_min, I_max, R_min, R_max, N,device)
        pred_nn_I_res_val = pred_nn_I_res_val.squeeze()
        val_pred.append(pred_nn_I_res_val)
    val_pred = torch.stack(val_pred)
    val_loss = criterion(val_pred, val_y)

    if val_loss - min_loss > min_delta:
        patience_count += 1
        if patience_count > patience:
            print(f"Early stopping at epoch {epoch + 1}...")
            break
    else:
        patience_count = 0
        min_loss = val_loss
    #     if val_loss<=min_loss:

    #         min_loss = val_loss
    print(f"Epoch {epoch + 1}: train loss = {loss:.4f}")

test_pred = []
for batch in range(test_x.shape[0]):
    I_max = I_max.reshape(I_max.shape[0])
    I_min = I_min.reshape(I_min.shape[0])
    pred_nn_I_res_test, pred_phy_I_res_test= model(test_x[batch], test_x_true[batch],
                                                                                    S_min, S_max, I_min, I_max, R_min,
                                                                                    R_max, N,device)
    pred_nn_I_res_test = pred_nn_I_res_test.squeeze()
    I_max = I_max.reshape(I_max.shape[0], 1)
    I_min = I_min.reshape(I_min.shape[0], 1)
    pred_nn_I_res_test_ = pred_nn_I_res_test * (I_max - I_min) + I_min

    test_pred.append(pred_nn_I_res_test_)
test_pred = torch.stack(test_pred)
mae_I = MAE(test_pred, test_y)
rmse_I = RMSE(test_pred, test_y)
mape_I = MAPE(test_pred, test_y)
pcc_I = PCC(test_pred, test_y)
ccc_I = CCC(test_pred, test_y)
print("I:(history_window:%d,pred_window:%d,slide_step:%d,MAE:%10f,RMSE:%f,MAPE:%f,PCC:%f,CCC:%f)" % (
history_window, pred_window, slide_step, mae_I, rmse_I, mape_I, pcc_I, ccc_I))