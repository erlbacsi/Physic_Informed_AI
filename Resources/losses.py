from typing import Any
import torch
from torch import nn

class RelativeLoss(object):
    def __init__(self, p=2):
        self.p = p
        
    def rel_loss(self, x, y):
        batch_size = x.size()[0]
        
        normed_diff = torch.norm(x.reshape(batch_size, -1) - y.reshape(batch_size, -1), self.p, 1)
        normed_y = torch.norm(y.reshape(batch_size, -1), self.p, 1)
        relative_loss = normed_diff / normed_y
        return torch.sum(relative_loss), torch.mean(relative_loss) 
    
    def z_score_normalize(self, x):
        mean_value = torch.mean(x)
        standard_deviation = torch.std(x)
        normalized = (x - mean_value) / (standard_deviation + 1e-8)
        return normalized
    
    def minMaxNorm(self, x):
        max_value = torch.max(x)
        min_value = torch.min(x)
        normed = (x - min_value) / (max_value - min_value)
        return normed
        
    def __call__(self, prediction, target):
        return self.rel_loss(prediction, target)


class PDERelative(nn.Module):
    def __init__(self, temperature_input=1):
        super(PDERelative, self).__init__()
        self.time_step_1 = 4
        self.time_step_2 = 40
        self.time_step_3 = 20
        self.indices = [i * 2 + 1 for i in range(temperature_input)]
        self.coord_dist = 0.2
        self.w = torch.tensor(2.5, dtype=torch.float32)
        self.w_optim = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.diffusivity = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = torch.tensor(1)
        self.rel_loss = RelativeLoss()
        
    def forward(self, x, prediction, target, time):
        temperature_input = x[:, self.indices, :]
        t_delta = torch.tensor(time).unsqueeze(1)
        t_delta = torch.where(t_delta < 280, torch.tensor(self.time_step_1, device="cuda:0"), t_delta)
        t_delta = torch.where(t_delta == 280, torch.tensor(self.time_step_2, device="cuda:0"), t_delta)
        t_delta = torch.where(t_delta >= 320, torch.tensor(self.time_step_3, device="cuda:0"), t_delta)
        
        output_grad = (prediction[:, :, 2:] - 2 * prediction[:, :, 1:-1] + prediction[:, :, :-2]) / (self.coord_dist ** 2) 
        first = output_grad[:, :, :1]
        last = output_grad[:, :, -1:]
        output_grad = torch.cat((first, output_grad, last), dim=2)
        output = temperature_input + t_delta * output_grad * self.diffusivity
        
        loss_sum, loss_mean = self.rel_loss.rel_loss(x=output, y=target)
        
        return loss_sum * self.alpha
    
    
class ResidualLoss(nn.Module):
    def __init__(self, temperature_input=1):
        super(ResidualLoss, self).__init__()
        self.time_step_1 = 4
        self.time_step_2 = 40
        self.time_step_3 = 20
        self.indices = [i * 2 + 1 for i in range(temperature_input)]
        self.w = torch.tensor(2.5, dtype=torch.float32)
        self.w_optim = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.diffusivity = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = torch.tensor(1)
        
    def z_score_normalize(self, x):
        mean_value = torch.mean(x)
        standard_deviation = torch.std(x)
        normalized = (x - mean_value) / (standard_deviation + 1e-8)
        return normalized
    
    def minMaxNorm(self, x):
        max_value = torch.max(x)
        min_value = torch.min(x)
        normed = (x - min_value) / (max_value - min_value)
        return normed

    def forward(self, x, prediction, time):
        diffusivity = self.diffusivity
        alpha = self.alpha 
        temperature = prediction
        temperature_input = x[:, self.indices, :]
        
        grad_x = torch.autograd.grad(outputs=temperature, inputs=x, grad_outputs=torch.ones_like(temperature), create_graph=True)[0]
        grad_grad_x = torch.autograd.grad(outputs=grad_x, inputs=x, grad_outputs=torch.ones_like(grad_x), create_graph=True)[0]
        grad_grad_x = grad_grad_x[:, 0, :].unsqueeze(1)
        
        grad_t = grad_x[:, -1, :].unsqueeze(1)
        
        residual = (grad_t - diffusivity * grad_grad_x) ** 2
        return torch.mean(residual) * alpha
        
class StefanLoss(torch.nn.Module):
    def __init__(self, temperature_input=4):
        super(StefanLoss, self).__init__()
        self.diff1 = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.time_step_1 = 4
        self.time_step_2 = 40
        self.time_step_3 = 20
        self.indices =  [i * 2 + 1 for i in range(temperature_input)]
        self.weight1 = 1
        self.weight2 = 0.00001
        self.w = torch.tensor(5, dtype=torch.float32)
        self.step_size = 0.2
        
        
    def forward(self, x, prediction_solid, prediction_fluid, time, interface, training=False):
        diff1 = torch.abs(self.diff1)
        temperature_solid = prediction_solid
        temperature_fluid = prediction_fluid
        
        temperature_input = x[:, self.indices, :]
            
        x_value = x[:, 0, :].unsqueeze(1)
        x_max = torch.max(x_value)
        x_min = torch.min(x_value)
        size = temperature_input.shape[-1] 
        
        # Output temperature picture
        ind_left, ind_right = self.get_nearest_index(interface=interface, max=x_max, min=x_min, size=size)
        temperature_combined = self.compute_target_temperature(temperature_fluid=temperature_fluid, temperature_solid=temperature_solid, ind_left=ind_left, ind_right=ind_right)
        
        if training:
            # Residual target
            grad_x = torch.autograd.grad(outputs=temperature_combined, inputs=x, grad_outputs=torch.ones_like(temperature_combined), create_graph=True)[0]
            grad_grad_x = torch.autograd.grad(outputs=grad_x, inputs=x, grad_outputs=torch.ones_like(grad_x), create_graph=True)[0]
            grad_grad_x = grad_grad_x[:, 0, :].unsqueeze(1)

            grad_t = grad_x[:, -1, :].unsqueeze(1)

            # grad_x = torch.autograd.grad(outputs=temperature_combined, inputs=x_value, grad_outputs=torch.ones_like(temperature_combined), create_graph=True)[0]
            # grad_grad_x = torch.autograd.grad(outputs=grad_x, inputs=x_value, grad_outputs=torch.ones_like(grad_x), create_graph=True)[0]
            # grad_t = torch.autograd.grad(outputs=temperature_solid, inputs=time, grad_outputs=torch.ones_like(temperature_solid), create_graph=True)[0]
            grad_tt, delta_t = self.compute_finite_difference(temperature_combined, temperature_input, time)
            residual_heat = torch.mean((grad_t - diff1 * grad_grad_x) ** 2) 

            # Interface Gradient
            temp_solid, temp_fluid = self.compute_phase_diff(temperature_solid=temperature_solid, temperature_fluid=temperature_fluid, ind_left=ind_left, ind_right=ind_right)
            temp_diff = torch.abs((temp_fluid / delta_t) - (temp_solid / delta_t))

            residual_interface = torch.mean((1 * (temp_fluid / self.step_size) - 1 * (temp_solid / self.step_size)  + interface / delta_t) ** 2)

            # Combined loss
            stefan_loss = self.weight1 * residual_heat + self.weight2 * residual_interface
            return stefan_loss, temperature_combined, residual_heat, residual_interface
        
        return temperature_combined
    
    def compute_finite_difference(self, output, input, delta):
        temperature_out = output
        temperature_in = input
        time = delta
        delta_t = time.detach().clone().unsqueeze(1)
        
        delta_t = torch.where(delta_t < 280, torch.tensor(self.time_step_1), delta_t)
        delta_t = torch.where(delta_t == 280, torch.tensor(self.time_step_2), delta_t)
        delta_t = torch.where(delta_t >= 320, torch.tensor(self.time_step_3), delta_t)
        
        grad_t = (temperature_out - temperature_in) / delta_t
        return grad_t, delta_t
    
    def compute_phase_diff(self, temperature_solid, temperature_fluid, ind_left, ind_right):
        # ind left hat immer linken index -> enthält linken index der linken grenze (solid) und linken index der rechten grenze (fluid) -> neu stacken nach solid und fluid
        ind_solid = torch.round(torch.stack((ind_left[:, :, 0], ind_right[:, :, 1]), dim=2)).to(dtype=torch.int64)
        ind_fluid = torch.round(torch.stack((ind_left[:, :, 1], ind_right[:, :, 0]), dim=2)).to(dtype=torch.int64)
        temp_solid = torch.gather(temperature_solid, 2, ind_solid)
        temp_fluid = torch.gather(temperature_fluid, 2, ind_fluid)
        # (batch_size, C, 2)
        return temp_solid, temp_fluid
        
    # normalisiere Koordinaten und transformiere sie auf maximalen Index
    def get_nearest_index(self, interface, max, min, size):
        interface = torch.where(interface < min, (min + self.step_size).detach().clone(), interface)
        interface = torch.where(interface > max, (max - self.step_size).detach().clone(), interface)
        gesamt = max - min
        transformed = ((interface - min) / gesamt) * (size - 1)
        ind = torch.round(transformed)
        ind_left = ind
        ind_left = torch.where(ind > transformed, ind - torch.tensor(1), ind_left)
        ind_right = ind
        ind_right = torch.where(ind < transformed, ind + torch.tensor(1), ind_right)
        
        left = torch.stack((ind_left[:, :, 0], ind_right[:, :, 0]), dim=2)
        right = torch.stack((ind_left[:, :, 1], ind_right[:, :, 1]), dim=2)
        return left, right
    
    
    def compute_target_temperature(self, temperature_solid, temperature_fluid, ind_left, ind_right):
        temperature_target = torch.zeros_like(temperature_fluid).to(device="cuda:0")
        left = ind_left[:, :, 0].unsqueeze(1)
        right = ind_right[:, :, 1].unsqueeze(1)
        
        indices = torch.arange(temperature_fluid.shape[2]).expand_as(temperature_fluid).to(device="cuda:0")
        # left area solid
        mask_left = indices < left
        temperature_left = temperature_solid * mask_left.to(device="cuda:0")
        temperature_target += temperature_left.to(device="cuda:0")

        mask_middle = (indices >= left) & (indices <= right)
        temperature_middle = temperature_fluid * mask_middle.to(device="cuda:0")
        temperature_target += temperature_middle.to(device="cuda:0")
        
        # right area solid
        mask_right = indices > right
        temperature_right = temperature_solid * mask_right.to(device="cuda:0")
        temperature_target += temperature_right.to(device="cuda:0")
        
        return temperature_target
        
        
        
        
        
        
        