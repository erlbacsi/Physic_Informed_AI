from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import degree
import numpy as np
import math
# import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)


#######################################################################################
# Fourier Neural Operator for 1D Input Data
#######################################################################################
class FourierConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, fourier_modes):
        super(FourierConvolution, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fourier_modes = fourier_modes
        self.scale = 1 / (in_channel + out_channel)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channel, out_channel, fourier_modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x)

        fourier_channels = x.shape[-1] // 2 + 1
        out_ft = torch.zeros(batchsize, self.out_channel, fourier_channels, device=x.device, dtype=torch.cfloat)
        # (batch, in_channel, x) * (in_channel, out_channel, x) -> (batch, out_channel, x)
        out_ft[:, :, :self.fourier_modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.fourier_modes], self.weights)

        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mid_channel, 1, dilation=dilation)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channel, out_channel, 1, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x


# Output from Fourier Convolution formed to same size as w for concatenation
# (if fourier conv in and out channels differ it is necessary)
class FNO(nn.Module):
    """
    input: (batch, channels, x)
    """
    def __init__(self, start_channels, out_channels, fourier_modes, width):
        super(FNO, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width
        self.drop_rate = 0.2
        self.encoder = nn.Linear(start_channels, self.width)
        # self.b1 = nn.BatchNorm1d(self.width)
        # self.b2 = nn.BatchNorm1d(self.width)
        # self.b3 = nn.BatchNorm1d(self.width)
        # self.b4 = nn.BatchNorm1d(self.width)
        self.norm = nn.InstanceNorm1d(self.width)
        # self.d1 = nn.Dropout1d(self.drop_rate)
        # self.d2 = nn.Dropout1d(self.drop_rate)
        # self.d3 = nn.Dropout1d(self.drop_rate)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        # self.form1 = Decoder(self.width, self.width, self.width, dilation=1)
        # self.form2 = Decoder(self.width, self.width, self.width, dilation=2)
        # self.form3 = Decoder(self.width, self.width, self.width, dilation=4)
        # self.form4 = Decoder(self.width, self.width, self.width, dilation=8)
        self.decoder = Decoder(self.width, self.width * 2, out_channels)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # x = self.b1(x)
        # after: (batch, channels, x)
        
        x1 = self.norm(self.fconv1(self.norm(x)))
        # x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        # x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        # x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        # x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)
        #
        x = self.decoder(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x

#######################################################################################


#######################################################################################
# Boundary Encoder to transform boundary Tensor to a higher dimension (output size)
#######################################################################################
class BoundaryEncoder(nn.Module):
    def __init__(self, input_channel, hidden_first, hidden_second, output_size):
        super(BoundaryEncoder, self).__init__()
        self.fc1 = nn.Linear(input_channel, hidden_first)
        self.fc2 = nn.Linear(hidden_first, hidden_second)
        self.fc3 = nn.Linear(hidden_second, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

#######################################################################################


#######################################################################################
# Stefan Fourier Neural Operator (input channels: coordinates, temperature, time)
#######################################################################################
class StefanFNO(nn.Module):
    def __init__(self, start_channels, out_channels, fourier_modes, width):
        super(StefanFNO, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width

        self.norm = nn.InstanceNorm1d(self.width)
        self.encoder = nn.Linear(start_channels, self.width)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        self.decoder = Decoder(self.width, self.width * 2, 2 * out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # after: (batch, channels, x)

        x1 = self.norm(self.fconv1(self.norm(x)))
        # x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        # x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        # x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        # x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x = self.decoder(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x
    
class InterfaceNetFNO(nn.Module):
    def __init__(self, start_channels, out_channels, interface_points, fourier_modes, width):
        super(InterfaceNetFNO, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width

        self.norm = nn.InstanceNorm1d(self.width)
        self.encoder = nn.Linear(start_channels, self.width)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        self.decoder = Decoder(self.width, self.width * 2, out_channels)
        # (batch_size, out_channels, x)
        # self.boundary = nn.AdaptiveAvgPool1d(interface_points)
        self.boundary = nn.AdaptiveMaxPool1d(interface_points)
        # (batch_size, out_channels, 2)

    def forward(self, x):
        max_coord = torch.max(x[0, 0, :])
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # after: (batch, channels, x)

        x1 = self.norm(self.fconv1(self.norm(x)))
        # x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        # x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        # x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        # x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x = self.decoder(x)
        x = self.boundary(x)
        # x = x.view(x.size(0), -1)
        
        x[:, 0, 1] = max_coord - x[:, 0, 1]
        
        # x = x.permute(0, 2, 1)
        return x
    
class InterfaceNetLinear(nn.Module):
    def __init__(self, input_channel, hidden1, hidden2, hidden3, output_channel=2):
        super(InterfaceNetLinear, self).__init__()
        self.time_step_1 = 4
        self.time_step_2 = 40
        self.time_step_3 = 20
        self.l1 = nn.Conv1d(input_channel, hidden1, 1)
        self.l2 = nn.Conv1d(hidden1, hidden2, 1)
        self.l3 = nn.Conv1d(hidden2, hidden3, 1)
        self.l4 = nn.Conv1d(hidden3, hidden3, 1)
        self.l5 = nn.Conv1d(hidden3, output_channel, 1)
        
    def forward(self, x, x_coord):
        max_coord = torch.max(x_coord[0, 0, :])
        
        times = x
        times = times.unsqueeze(-1)
        times = torch.where(times < 280, times + torch.tensor(self.time_step_1), times)
        times = torch.where(times == 280, times + torch.tensor(self.time_step_2), times)
        times = torch.where(times > 280, times + torch.tensor(self.time_step_3), times)
        
        # (batch_size, C, 1)
        
        x = self.l1(times)
        x = F.tanh(x)
        x = self.l2(x)
        x = F.tanh(x)
        x = self.l3(x)
        x = F.tanh(x)
        x = self.l4(x)
        x = F.tanh(x)
        x = self.l5(x)
        x = x.permute(0, 2, 1)
        
        x[:, 0, 1] = max_coord - x[:, 0, 1]
        # (batch_size, C, 2)
        return x
        
        
#######################################################################################


#######################################################################################
# FNO hybrid with and without Fourier Continuation
#######################################################################################
class FNObasic(nn.Module):
    def __init__(self, start_channels, out_channels, fourier_modes, width):
        super(FNObasic, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width
        self.drop_rate = 0.2
        self.encoder = nn.Linear(start_channels, self.width)
        # self.b1 = nn.BatchNorm1d(self.width)
        # self.b2 = nn.BatchNorm1d(self.width)
        # self.b3 = nn.BatchNorm1d(self.width)
        # self.b4 = nn.BatchNorm1d(self.width)
        self.norm = nn.InstanceNorm1d(self.width)
        # self.d1 = nn.Dropout1d(self.drop_rate)
        # self.d2 = nn.Dropout1d(self.drop_rate)
        # self.d3 = nn.Dropout1d(self.drop_rate)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        # self.form1 = Decoder(self.width, self.width, self.width, dilation=1)
        # self.form2 = Decoder(self.width, self.width, self.width, dilation=2)
        # self.form3 = Decoder(self.width, self.width, self.width, dilation=4)
        # self.form4 = Decoder(self.width, self.width, self.width, dilation=8)
        self.decoder = Decoder(self.width, self.width * 2, out_channels)
        self.left_b = torch.tensor([0.25], dtype=torch.float32).to(device="cuda:0")
        self.right_b = torch.tensor([0.75], dtype=torch.float32).to(device="cuda:0")
    
    def set_nearest_index(self, interface):
        min_left = torch.tensor([0.01]).to(device="cuda:0")
        max_right = torch.tensor([0.99]).to(device="cuda:0")
        interface[:, :, 1] = 1.0 - interface[:, :, 1]
        
        interface[:, :, 0] = torch.where(interface[:, :, 0] <= min_left, min_left, interface[:, :, 0])
        interface[:, :, 1] = torch.where(interface[:, :, 1] >= max_right, max_right, interface[:, :, 1])
        
        self.left_b = interface[:, :, 0].unsqueeze(1)
        self.right_b = interface[:, :, 1].unsqueeze(1)
        
        # prozent in Koordinaten umwandeln
        # interface *= max
        
        # interface = torch.where(interface < min, torch.tensor(min + self.step_size), interface)
        # interface = torch.where(interface > max, torch.tensor(max - self.step_size), interface)
        # gesamt = max - min
        # transformed = ((interface - min) / gesamt) * (size - 1)
        # ind = torch.round(transformed)
        # ind_left = ind
        # ind_left = torch.where(ind > transformed, ind - torch.tensor(1), ind_left)
        # ind_right = ind
        # ind_right = torch.where(ind < transformed, ind + torch.tensor(1), ind_right)
        # return ind_left, ind_right
    
    def compute_target_temperature(self, temperature, temperature_c, size, time):
        # left = torch.where(time <= torch.tensor([40], dtype=torch.float32).to(device="cuda:0"), torch.tensor([0.1], dtype=torch.float32).to(device="cuda:0"), torch.tensor([0.25], dtype=torch.float32).to(device="cuda:0")).unsqueeze(dim=1)
        # right = torch.where(time <= torch.tensor([40], dtype=torch.float32).to(device="cuda:0"), torch.tensor([0.9], dtype=torch.float32).to(device="cuda:0"), torch.tensor([0.75], dtype=torch.float32).to(device="cuda:0")).unsqueeze(dim=1)
        # 
        # left = torch.round(size * left)
        # right = torch.round(size * right)
        
        left = torch.round(size * self.left_b)
        right = torch.round(size * self.right_b)
        
        temperature_target = torch.zeros_like(temperature).to(device="cuda:0")
        # if left.shape[0] == 1:
        #     left = left.repeat(temperature.shape[0], 1, 1)
        #     right = right.repeat(temperature.shape[0], 1, 1)
        # else:
        #     left = left[:, :, 0].unsqueeze(1)
        #     right = right[:, :, 1].unsqueeze(1)
        
        indices = torch.arange(temperature.shape[2]).expand_as(temperature).to(device="cuda:0")
        # left area solid
        mask_left = indices < left
        temperature_left = temperature_c * mask_left.to(device="cuda:0")
        temperature_target += temperature_left.to(device="cuda:0")  
        
        # middle area
        mask_middle = (indices >= left) & (indices <= right)
        temperature_middle = temperature * mask_middle.to(device="cuda:0")
        temperature_target += temperature_middle.to(device="cuda:0")
        
        # right area solid
        mask_right = indices > right
        temperature_right = temperature_c * mask_right.to(device="cuda:0")
        temperature_target += temperature_right.to(device="cuda:0")
        
        return temperature_target
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # x = self.b1(x)
        # after: (batch, channels, x)
        
        # x1 = self.norm(self.fconv1(self.norm(x)))
        x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = self.d1(x)
        
        # x1 = self.norm(self.fconv2(self.norm(x)))
        x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = self.d2(x)
        
        # x1 = self.norm(self.fconv3(self.norm(x)))
        x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = self.d3(x)
        
        # x1 = self.norm(self.fconv4(self.norm(x)))
        x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.decoder(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x

class InterfaceNetFNOHybrid(nn.Module):
    def __init__(self, start_channels, out_channels, interface_points, fourier_modes, width):
        super(InterfaceNetFNOHybrid, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width
        
        self.norm = nn.InstanceNorm1d(self.width)

        self.encoder = nn.Linear(start_channels, self.width)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        self.decoder = Decoder(self.width, self.width * 2, out_channels)
        # (batch_size, out_channels, x)
        self.boundary = nn.AdaptiveAvgPool1d(interface_points)
        # (batch_size, out_channels, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # x = self.b1(x)
        # after: (batch, channels, x)
        
        x1 = self.norm(self.fconv1(self.norm(x)))
        # x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        # x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        # x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        # x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.decoder(x)
        
        x = self.boundary(x)
        # Clip values between 0.0 and 1.0
        x = F.sigmoid(x)
        # x = x.view(x.size(0), -1)
        
        # x = x.permute(0, 2, 1)
        return x
    
#######################################################################################


#######################################################################################
# Laplace Neural Operator
#######################################################################################
class LaplaceConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, modes):
        super(LaplaceConvolution, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.modes = modes
        self.scale = 1 / (in_channel + out_channel)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channel, out_channel, modes, dtype=torch.cfloat))

    def forward(self, x, pseudo_inverse, eigenvector):
        batchsize = x.shape[0]
 
        x_spectral = x @ pseudo_inverse.to(device="cuda:0", dtype=torch.float32)
        x_spectral = x_spectral.to(dtype=torch.cfloat)
        
        laplace_channels = x.shape[-1] 
        out_spectral = torch.zeros(batchsize, self.out_channel, laplace_channels, device=x.device, dtype=torch.cfloat)
        # (batch, in_channel, x) * (in_channel, out_channel, x) -> (batch, out_channel, x)
        out_spectral[:, :, :self.modes] = torch.einsum("bix,iox->box", x_spectral[:, :, :self.modes], self.weights)

        x = out_spectral @ eigenvector.to(device="cuda:0", dtype=torch.cfloat)
        x = x.to(dtype=torch.float32)
        
        return x

class DecoderLaplace(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation=1):
        super(DecoderLaplace, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mid_channel, 1, dilation=dilation)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channel, out_channel, 1, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x
    
class LNO(nn.Module):
    def __init__(self, start_channels, out_channels, modes, width):
        super(LNO, self).__init__()
        self.start_channels = start_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.encoder = nn.Linear(start_channels, self.width)
        self.norm = nn.InstanceNorm1d(self.width)
        self.lconv1 = LaplaceConvolution(self.width, self.width, self.modes)
        self.lconv2 = LaplaceConvolution(self.width, self.width, self.modes)
        self.lconv3 = LaplaceConvolution(self.width, self.width, self.modes)
        self.lconv4 = LaplaceConvolution(self.width, self.width, self.modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = DecoderLaplace(self.width, self.width, self.width)
        self.form2 = DecoderLaplace(self.width, self.width, self.width)
        self.form3 = DecoderLaplace(self.width, self.width, self.width)
        self.form4 = DecoderLaplace(self.width, self.width, self.width)
        self.decoder = DecoderLaplace(self.width, self.width * 2, out_channels)
        self.time_step_1 = 4
        self.time_step_2 = 40
        self.time_step_3 = 20
        # number of temperature channels
        self.indices = [i * 2 + 1 for i in range(start_channels-2)]
        # kann auch immer aus den Daten bestimmt werden und kann beliebig sein -> mesh invariant
        self.coord_dist = 0.2
        # self.pseudo_inverse, self.eigenvector = self.laplacian_computation(self.size)
    
    def laplacian_computation(self, x):
        size = x.shape[-1]
        laplacian = (torch.diag(torch.ones(size-1), -1) - 2 * torch.diag(torch.ones(size), 0) + torch.diag(torch.ones(size-1), 1)) / (self.coord_dist**2)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # eigenvectors = eigenvectors[, :self.modes]
        
        pseudo_inverse = (eigenvectors.T @ eigenvectors).inverse() @ eigenvectors.T
        return pseudo_inverse, eigenvectors
    
        
    def forward(self, x):
        # transformation vorschfrift für Spektralbereich
        pseudo_inverse, eigenvector = self.laplacian_computation(x)
        
        x = x.permute(0, 2, 1)
        # after: (batch, x, channels)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        # x = self.b1(x)
        # after: (batch, channels, x)
        
        # x1 = self.norm(self.lconv1(self.norm(x)))
        x1 = self.norm(self.lconv1(self.norm(x), pseudo_inverse, eigenvector))
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d1(x)
        
        # x1 = self.norm(self.lconv2(self.norm(x)))
        x1 = self.norm(self.lconv2(self.norm(x), pseudo_inverse, eigenvector))
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d2(x)
        
        # x1 = self.norm(self.lconv3(self.norm(x)))
        x1 = self.norm(self.lconv3(self.norm(x), pseudo_inverse, eigenvector))
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d3(x)
        
        # x1 = self.norm(self.lconv4(self.norm(x)))
        x1 = self.norm(self.lconv4(self.norm(x), pseudo_inverse, eigenvector))
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x = self.decoder(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x

#######################################################################################


#######################################################################################
# Geometry Informed Neural Operator (GINO)
#######################################################################################
class GNO(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(GNO, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        
        # self.linear = nn.Linear(in_channels, mid_channels)
        self.gcn1 = GCNConv(self.in_channels, self.mid_channels)
        self.gcn2 = GCNConv(self.mid_channels, self.mid_channels)
        self.gcn3 = GCNConv(self.mid_channels, self.mid_channels)
        self.gcn4 = GCNConv(self.mid_channels, self.out_channels)
        self.layerNorm1 = nn.LayerNorm(in_channels)
        self.layerNorm2 = nn.LayerNorm(mid_channels)
        self.layerNorm3 = nn.LayerNorm(mid_channels)
        self.layerNorm4 = nn.LayerNorm(mid_channels)
        
    def forward(self, x, edge_index):
        # x = self.linear(x)
        x = self.gcn1(self.layerNorm1(x), edge_index)
        x = F.gelu(x)
        x = self.gcn4(self.layerNorm4(x), edge_index)
        return x
    
class TemperatureLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(TemperatureLayer, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.temp_diff = torch.nn.Linear(1, 1)
        self.gcn = torch.nn.Linear(self.in_channels, self.out_channels)
    
    def forward(self, x, edge_index): 
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index=edge_index, x=x, norm=norm)
        
    def message(self, x_i, x_j):
        x_diff = (x_j[:, 1] - x_i[:, 1]).unsqueeze(1)
        #x_diff = self.temp_diff(x_diff)
        #x_diff = F.gelu(x_diff)
        return x_diff
    
    def update(self, aggr_out, x):
        updated_features = torch.cat([x[:, 0].unsqueeze(1), aggr_out], dim=1)
        updated_features = torch.cat([updated_features, x[:, -1].unsqueeze(1), ], dim=1)
        return self.gcn(updated_features)
    
class CollectionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CollectionLayer, self).__init__(aggr="mean") 
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.gcn = torch.nn.Linear(self.in_channels, self.out_channels)
        # self.gcn2 = torch.nn.Linear(2 * self.out_channels, self.out_channels)
    
    def forward(self, x, edge_index):
        x = self.gcn(x)
        # x = F.gelu(x)
        
        #row, col = edge_index
        #deg = degree(col, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        #return self.propagate(edge_index=edge_index, x=x, norm=norm)
        return self.propagate(edge_index=edge_index, x=x)
        
    def message(self, x_i, x_j):
        # m = torch.cat([x_i, x_j], dim=1)
        # x = self.gcn2(m)
        return x_j
        
class EdgeConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvLayer, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.gcn1 = torch.nn.Linear(2 * self.in_channels, self.out_channels)
        #self.gcn1 = torch.nn.Linear(self.in_channels, self.out_channels)
        self.gcn2 = torch.nn.Linear(self.out_channels, self.out_channels)
    
    def forward(self, x, edge_index):
        # Compute normalization.
        #row, col = edge_index
        #deg = degree(col, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index=edge_index, x=x)
        
    def message(self, x_i, x_j):
        m = torch.cat([x_i, x_j - x_i], dim=1)
        # m = x_j - x_i
        # m = self.gcn1(m)
        # m = F.gelu(m)
        return self.gcn1(m)

class GNOCustomEncode(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(GNOCustomEncode, self).__init__()
        self.temp = TemperatureLayer(in_channels=in_channels, out_channels=mid_channels)
        self.collect2 = CollectionLayer(in_channels=mid_channels, out_channels=mid_channels)
        self.collect1 = CollectionLayer(in_channels=in_channels, out_channels=mid_channels)
        self.collect3 = CollectionLayer(in_channels=mid_channels, out_channels=out_channels)
        
        self.edge1 = EdgeConvLayer(in_channels=in_channels, out_channels=mid_channels)
        self.edge2 = EdgeConvLayer(in_channels=mid_channels, out_channels=mid_channels)
        
        self.layerNorm1 = nn.LayerNorm(in_channels)
        self.layerNorm2 = nn.LayerNorm(mid_channels)
        
    def forward(self, x, edge_index):
        x = self.edge1(x, edge_index)
        x = F.gelu(x)
        x = self.collect3(x, edge_index)
        #x = F.gelu(x)
        #x = self.edge2(x, edge_index)
        #x = F.gelu(x)
        #x = self.collect3(x, edge_index)
        return x

class GNOCustomDecode(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(GNOCustomDecode, self).__init__()
        self.collect1 = CollectionLayer(in_channels=in_channels, out_channels=mid_channels)
        self.collect2 = CollectionLayer(in_channels=mid_channels, out_channels=mid_channels)
        self.collect3 = CollectionLayer(in_channels=mid_channels, out_channels=out_channels)
          
        self.edge1 = EdgeConvLayer(in_channels=in_channels, out_channels=mid_channels)
        self.edge2 = EdgeConvLayer(in_channels=mid_channels, out_channels=mid_channels)
        
        self.layerNorm1 = nn.LayerNorm(in_channels)
        self.layerNorm2 = nn.LayerNorm(mid_channels)
        
    def forward(self, x, edge_index):
        x = self.collect1(x, edge_index)
        x = F.gelu(x)
        x = self.collect3(x, edge_index)
        #x = F.gelu(x)
        #x = self.edge2(x, edge_index)
        #x = F.gelu(x)
        #x = self.collect3(x, edge_index)
        return x

class GINO(nn.Module):
    def __init__(self, start_channels, out_channels, fourier_modes, width, radius):
        super(GINO, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes = fourier_modes
        self.width = width
        self.radius = radius
        
        self.encoder = GNOCustomEncode(in_channels=start_channels, mid_channels=width, out_channels=width)  
              
        self.norm = nn.InstanceNorm1d(self.width)
        self.fconv1 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv2 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv3 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.fconv4 = FourierConvolution(self.width, self.width, self.fourier_modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.form1 = Decoder(self.width, self.width, self.width)
        self.form2 = Decoder(self.width, self.width, self.width)
        self.form3 = Decoder(self.width, self.width, self.width)
        self.form4 = Decoder(self.width, self.width, self.width)
        
        self.decoder = GNOCustomDecode(in_channels=self.width, mid_channels=self.width * 2, out_channels=out_channels)
        #self.decoder2 = Decoder(self.width, self.width, out_channels)
        

    def forward(self, x):
        size = x.shape[-1]
        batch_size = x.shape[0]
        pos = x[:, 0, :].unsqueeze(dim=1)
        pos = pos.permute(0, 2, 1)
        channel = pos.shape[-1]
        # (batch_size*x, channel) -> geometric erwartet (num_points, num_dimensions)
        pos = pos.reshape(-1, channel).to("cpu")
        # Batch Index zuordnen den inputs von oben (0, 0, 0, 0, 1, 1, 1, 1,....)
        batch = torch.arange(batch_size).repeat_interleave(size).to("cpu")
        edge_index = radius_graph(x=pos, r=self.radius, batch=batch, loop=True)
        edge_index = edge_index.to(device="cuda:0")
        
        x = x.permute(0, 2, 1)
        channels = x.shape[-1]
        x = x.reshape(-1, channels)
        # after: (batch_size * num_points, channels)
        x = self.encoder(x, edge_index)
        x = x.reshape(batch_size, -1, x.shape[1])
        x = x.permute(0, 2, 1)
        # after: (batch, channels, x)

        x1 = self.norm(self.fconv1(self.norm(x)))
        # x1 = self.fconv1(x)
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        # x1 = self.fconv2(x)
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        # x1 = self.fconv3(x)
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = F.tanh(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        # x1 = self.fconv4(x)
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x.permute(0, 2, 1)
        channels = x.shape[-1]
        x = x.reshape(-1, channels)
        x = self.decoder(x, edge_index)
        x = x.reshape(batch_size, -1, x.shape[1])
        x = x.permute(0, 2, 1)
        
        # x = F.gelu(x)
        # x = self.decoder2(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x
#######################################################################################


#######################################################################################
# Laplace Fourier Attention
#######################################################################################
class LFA(nn.Module):
    def __init__(self, start_channels, out_channels, fourier_modes, laplace_modes, width):
        super(LFA, self).__init__()
        
        self.fno = FNO(start_channels=start_channels, out_channels=out_channels, fourier_modes=fourier_modes, width=width)
        self.lno = LNO(start_channels=start_channels, out_channels=out_channels, modes=laplace_modes, width=width)
        
        self.positional_encoding = PositionalEncoding1D(width)
        
        self.attention1 = nn.MultiheadAttention(embed_dim=width, num_heads=8)
        #self.attention2 = nn.MultiheadAttention(embed_dim=width, num_heads=4)
        
        self.aggregate = Decoder(in_channel=out_channels, mid_channel=width, out_channel=width)
        # self.aggregate = LinearDecoder(in_features=out_channels, hidden=width)
        # self.layerNorm = nn.LayerNorm(width)
        #self.norm = nn.InstanceNorm1d(2)
        #self.norm_ = nn.InstanceNorm1d(width)
        
        self.layerNorm1 = nn.LayerNorm(width)
        self.layerNorm1ff = nn.LayerNorm(width)
        #self.layerNorm2 = nn.LayerNorm(width)
        #self.layerNorm2ff = nn.LayerNorm(width)
       
        # self.decoder1 = Decoder(in_channel=width, mid_channel=width*2, out_channel=width)
        # self.decoder2 = Decoder(in_channel=width, mid_channel=width*2, out_channel=width)
        self.decoder_out = Decoder(in_channel=width, mid_channel=width*2, out_channel=out_channels)
        
        self.ff1 = FeedForward(width, width)
        #self.ff2 = FeedForward(width, width)
        

    def forward(self, x):
        x_fno = self.fno(x)
        #x_fno = self.lno(x)
        
        #x = torch.cat([x_fno, x_lno], dim=1)
        x = self.aggregate(x_fno)
        # (batch_size, c, x)
        
        # x = x.permute(1, 0, 2)
        x = x.permute(2, 0, 1)
        # positional encoding
        ## (x, batch_size, c)
        x = self.positional_encoding(x)
        
        # attention block 1
        x_residual = x
        x, _ = self.attention1(value=x, key=x, query=x)
        x = self.layerNorm1(x + x_residual)
        
        x_residual = x
        x = self.ff1(x)
        x = self.layerNorm1ff(x + x_residual)
        
        # attention block 2
        #x_residual = x
        #x, _ = self.attention2(value=x, key=x, query=x)
        #x = self.layerNorm2(x + x_residual)
    
        #x_residual = x
        #x = self.ff2(x)
        #x = self.layerNorm2ff(x + x_residual)
        
        # Decode
        x = x.permute(1, 2, 0)
        x = self.decoder_out(x)
        
        return x
    
    
    '''def old_forward__(self, x):
        x_fno = self.fno(x)
        x_lno = self.lno(x)
        x = torch.cat([x_fno, x_lno], dim=1)
        x = self.aggregate(x)
        # (batch_size, c, x)
        x = x.permute(2, 0, 1)
        #x = self.layerNorm(x)
        
        # attention block 1
        x = self.positional_encoding(x)
        
        x_residual = x
        x, _ = self.attention1(value=x, key=x, query=x)
        x = self.layerNorm1(x + x_residual)
        
        x_residual = x
        x = x.permute(1, 2, 0)
        x = self.decoder_(x)
        x = x.permute(2, 0, 1)
        self.layerNorm1ff(x + x_residual)
        
        # attention block 2
        x_residual = x
        x, _ = self.attention2(value=x, key=x, query=x)
        #x = self.dropout1(x)
        x = self.layerNorm2(x + x_residual)
        
        x_residual = x
        x = self.ff2(x)
        x = self.layerNorm2ff(x + x_residual)
    
        ## attention block 3
        #x_residual = x
        #x, _ = self.attention3(value=x, key=x, query=x)
        #x = self.layerNorm3(x + x_residual)
        
        #x_residual = x
        #x = x.permute(1, 2, 0)
        #x = self.decoder3(x)
        #x = x.permute(2, 0, 1)
        #x = self.layerNorm3ff(x + x_residual)
        
        x = x.permute(1, 2, 0)
        x = self.decoder_out(x)
        
        return x'''

class FeedForward(nn.Module):
    def __init__(self, in_features, hidden):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(in_features=in_features, out_features=hidden*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.l2 = nn.Linear(in_features=hidden*2, out_features=hidden)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x
    
class LinearDecoder(nn.Module):
    def __init__(self, in_features, hidden):
        super(LinearDecoder, self).__init__()
        self.l1 = nn.Linear(in_features=in_features, out_features=hidden)
        self.gelu = nn.ReLU()
        self.l2 = nn.Linear(in_features=hidden, out_features=hidden)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super(PositionalEncoding1D, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        # Batch dimension einfügen
        self.encoding = self.encoding.unsqueeze(1).to(device="cuda:0")

    def forward(self, x):
        return x + self.encoding[:x.size(0), :, :]
    
        
#######################################################################################



#######################################################################################
#
#######################################################################################
class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self):
        pass
#######################################################################################


#######################################################################################
#
#######################################################################################
class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self):
        pass
#######################################################################################





#######################################################################################
# Fourier Neural Operator for 2D Input Data
#######################################################################################
class FourierConvolution2D(nn.Module):
    def __init__(self, in_channel, out_channel, fourier_modes_1, fourier_modes_2):
        super(FourierConvolution2D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fourier_modes_1 = fourier_modes_1
        self.fourier_modes_2 = fourier_modes_2
        self.scale = 1 / (in_channel + out_channel)
        self.weights_1 = nn.Parameter(self.scale * torch.rand(in_channel, out_channel, fourier_modes_1, fourier_modes_2, dtype=torch.cfloat))
        self.weights_2 = nn.Parameter(self.scale * torch.rand(in_channel, out_channel, fourier_modes_1, fourier_modes_2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        fourier_channels = x.shape[-1] // 2 + 1
        out_ft = torch.zeros(batchsize, self.out_channel, x.shape[-2], fourier_channels, device=x.device, dtype=torch.cfloat)
        # (batch, in_channel, x) * (in_channel, out_channel, x) -> (batch, out_channel, x)
        out_ft[:, :, :self.fourier_modes_1, :self.fourier_modes_2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.fourier_modes_1, :self.fourier_modes_2], self.weights_1)
        out_ft[:, :, -self.fourier_modes_1:, :self.fourier_modes_2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.fourier_modes_1:, :self.fourier_modes_2], self.weights_2)

        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class Decoder2D(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, dilation=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 1, dilation=dilation)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 1, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x


# Output from Fourier Convolution formed to same size as w for concatenation
# (if fourier conv in and out channels differ it is necessary)
class FNO2D(nn.Module):
    """
    input: (batch, channels, x)
    """
    def __init__(self, start_channels, out_channels, fourier_modes_1, fourier_modes_2, width):
        super(FNO2D, self).__init__()
        self.start_channels = start_channels
        self.fourier_modes_1 = fourier_modes_1
        self.fourier_modes_2 = fourier_modes_2
        self.width = width
        self.drop_rate = 0.2
        self.encoder = nn.Linear(start_channels, self.width)
        # self.b1 = nn.BatchNorm1d(self.width)
        # self.b2 = nn.BatchNorm1d(self.width)
        # self.b3 = nn.BatchNorm1d(self.width)
        # self.b4 = nn.BatchNorm1d(self.width)
        # self.d1 = nn.Dropout1d(self.drop_rate)
        # self.d2 = nn.Dropout1d(self.drop_rate)
        # self.d3 = nn.Dropout1d(self.drop_rate)
        self.norm = nn.InstanceNorm2d(self.width)
        self.fconv1 = FourierConvolution2D(self.width, self.width, self.fourier_modes_1, self.fourier_modes_2)
        self.fconv2 = FourierConvolution2D(self.width, self.width, self.fourier_modes_1, self.fourier_modes_2)
        self.fconv3 = FourierConvolution2D(self.width, self.width, self.fourier_modes_1, self.fourier_modes_2)
        self.fconv4 = FourierConvolution2D(self.width, self.width, self.fourier_modes_1, self.fourier_modes_2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.form1 = Decoder2D(self.width, self.width, self.width)
        self.form2 = Decoder2D(self.width, self.width, self.width)
        self.form3 = Decoder2D(self.width, self.width, self.width)
        self.form4 = Decoder2D(self.width, self.width, self.width)
        # self.form1 = Decoder(self.width, self.width, self.width, dilation=1)
        # self.form2 = Decoder(self.width, self.width, self.width, dilation=2)
        # self.form3 = Decoder(self.width, self.width, self.width, dilation=4)
        # self.form4 = Decoder(self.width, self.width, self.width, dilation=8)
        self.decoder = Decoder2D(self.width, self.width * 2, out_channels)
        
    def forward(self, x):
        # (batch, c, y, x)
        x = x.permute(0, 2, 3, 1)
        # after: (batch, x, y, channels)
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        # x = self.b1(x)
        # after: (batch, channels, y, x)
        
        x1 = self.norm(self.fconv1(self.norm(x)))
        x1 = self.form1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.b2(x)
        x = F.gelu(x)
        # x = self.d1(x)
        
        x1 = self.norm(self.fconv2(self.norm(x)))
        x1 = self.form2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        # x = self.b3(x)
        x = F.gelu(x)
        # x = self.d2(x)
        
        x1 = self.norm(self.fconv3(self.norm(x)))
        x1 = self.form3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.b4(x)
        x = F.gelu(x)
        # x = self.d3(x)
        
        x1 = self.norm(self.fconv4(self.norm(x)))
        x1 = self.form4(x1)
        x2 = self.w4(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.decoder(x)
        
        # x = x[:, :, :5001]
        # x = x.permute(0, 2, 1)
        return x
    
#######################################################################################




if __name__ == "__main__":
    data = torch.rand([10, 3, 251])
    data[:, 0, 2] = 1111
    gino = GINO(3, 1, 16, 64, 0.3)
    x = gino(data)
    print(x.shape)


