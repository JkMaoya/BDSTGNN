
from torch.nn.utils import weight_norm


import torch
from torch import nn
import torch.nn.functional as F
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out


class GCN_layer(nn.Module):
    def __init__(self, in_dim_gcn, out_dim_gcn):
        super(GCN_layer, self).__init__()
        self.in_dim_gcn = in_dim_gcn
        self.out_dim_gcn = out_dim_gcn
        self.weight = nn.Parameter(torch.randn(5, in_dim_gcn, out_dim_gcn))

    def forward(self, adj, data):
        support = torch.einsum('tnw,wtd->wtd', (adj, data))

        output = torch.einsum('wtd, tdo -> wto', support, self.weight)
        return output


class GCN_Model(nn.Module):
    def __init__(self, in_dim_gcn, out_dim_gcn):
        super(GCN_Model, self).__init__()
        self.GCN1 = GCN_layer(in_dim_gcn, out_dim_gcn)

        self.relu = nn.ReLU()

    def forward(self, adj, data):
        gcnlayer1 = self.GCN1(adj, data)

        return gcnlayer1


class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class BDSTGNN(nn.Module):
    def __init__(self, node_embed, kernel_size, out_dim_gcn
                 , history_window, pred_window, num_nodes, num_channels_graph, kernel_size_graph, dropout):
        super(BDSTGNN, self).__init__()

        self.GCN1 = GCN_Model(node_embed, out_dim_gcn)
        self.GCN2 = GCN_Model(out_dim_gcn, out_dim_gcn)
        self.x_emd = nn.Linear(3, node_embed)
        self.fc_I = nn.Linear(3, node_embed)

        self.decom_I = series_decomp(kernel_size)

        self.fc_I_res = nn.Linear(node_embed, node_embed)
        self.fc_I_trend = nn.Linear(node_embed, node_embed)

        self.sigmoid = nn.Sigmoid()
        self.history_window = history_window

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)

        nn.init.xavier_uniform_(self.nodevec1)

        self.result = nn.Linear(5 * out_dim_gcn, pred_window)
        self.num_nodes = num_nodes
        self.intra = nn.Linear(out_dim_gcn * history_window, 2)
        self.pred_window = pred_window
        self.TCN_adj1 = TemporalConvNet(node_embed, num_channels_graph, kernel_size_graph, dropout)

    def forward(self, x, data_true, s_min, s_max, i_min, i_max, r_min, r_max, N,device):
        I_emd = self.fc_I(x)

        I_res, I_trend = self.decom_I(I_emd)

        I_res_trans = self.fc_I_res(I_res)

        I_trend_trans = self.fc_I_trend(I_trend)

        I_tem = I_res_trans + I_trend_trans
        all_adj_norm = torch.ones(5, self.num_nodes, self.num_nodes)

        nodevec = self.nodevec1

        sta_adj = torch.mm(nodevec, nodevec.transpose(0, 1))
        sta_adj = F.softmax(F.relu(sta_adj), dim=1)

        dyn_emd1 = self.TCN_adj1(I_emd.permute(0, 2, 1))
        dyn_emd1 = dyn_emd1.permute(2, 0, 1)
        dyn_emd2 = dyn_emd1.permute(0, 2, 1)

        dyn_adj = torch.matmul(dyn_emd1, dyn_emd2)
        dyn_adj = F.softmax(F.relu(dyn_adj), dim=2)
        fus_adj = dyn_adj + sta_adj
        fus_adj = F.softmax(F.relu(fus_adj), dim=2)
        all_adj_norm = all_adj_norm * fus_adj

        gcn_res2 = self.GCN1(all_adj_norm, I_tem)
        intra_para = self.intra(gcn_res2.reshape(self.num_nodes, -1))
        beta_mean = intra_para[:, 0]
        gama_mean = intra_para[:, 1]
        beta_mean = torch.sigmoid(beta_mean)
        gama_mean = torch.sigmoid(gama_mean)

        last_S_out = torch.zeros([self.num_nodes, self.pred_window + 1])
        last_S_out[:, 0] = data_true[:, -1, 2]
        last_I_out = torch.zeros([self.num_nodes, self.pred_window + 1])
        last_I_out[:, 0] = data_true[:, -1, 0]
        last_R_out = torch.zeros([self.num_nodes, self.pred_window + 1])
        last_R_out[:, 0] = data_true[:, -1, 1]
        pred_phy_I_res = torch.tensor([]).to(device)
        for seq in range(self.pred_window):
            phy_cur_loc_I = []

            dI_ = (beta_mean[:] * last_S_out[:, seq].clone() / N[:] - gama_mean[:]) * last_I_out[:, seq].clone()
            dR_ = gama_mean[:] * last_I_out[:, seq].clone()

            last_I_out[:, seq + 1] = last_I_out[:, seq] + dI_
            last_R_out[:, seq + 1] = last_R_out[:, seq] + dR_
            last_S_out[:, seq + 1] = N[:] - last_I_out[:, seq + 1] - last_R_out[:, seq + 1]

            I_num_phy = (last_I_out[:, seq + 1].clone() - i_min[:]) / (i_max[:] - i_min[:]).unsqueeze(0)

            phy_cur_loc_I = I_num_phy
            pred_phy_I_res = torch.cat([pred_phy_I_res, phy_cur_loc_I.to(torch.float32)], dim=0)

        pred_phy_I_res = pred_phy_I_res.permute(1, 0)

        final_res = self.result((gcn_res2).reshape(self.num_nodes, -1))
        final_res = final_res
        return final_res, pred_phy_I_res