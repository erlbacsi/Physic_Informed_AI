import os
import numpy as np
import pandas as pd
import torch
import torch.onnx
from torch.utils.data import DataLoader 
from torch import nn
from Resources.early_stopping import EarlyStopping
from Resources.losses import RelativeLoss, ResidualLoss, PDERelative
from Resources.temp_dataset import SingleStepDataset1D, SingleStepMultiTempDataset1D, SingleStepMultiTempDataset1Dtesting, MultiStepMultiTempDataset1D
from Resources.ai_models import FNO
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import onnxruntime as ort
from torch.utils.tensorboard import SummaryWriter
import datetime
from timeit import default_timer
from sklearn.model_selection import train_test_split


print(torch.__version__)
print(torch.version.cuda)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')
  
absolute_path = os.getcwd()
relative_path = "logs/thermal_check.pth"
path = os.path.join(absolute_path, relative_path)
print(path)


def save_checkpoint(model1, model2, optimizer, epoch, train_loss, val_loss, t_max_diff, t_min_diff, v_max_diff, v_min_diff, time, save_path):
    if model2 is None:
        torch.save({
            'model1': model1.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)
    else:
        torch.save({
            "model1": model1.state_dict(),
            "model2": model2.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

    diction = {'Train Loss':train_loss, 'Validation Loss':val_loss, 'Training Max Diff': t_max_diff, 'Training Min Diff':t_min_diff, 'Validation Max Diff': v_max_diff, 'Validation Min Diff':v_min_diff, "Execution Time": time}
    df = pd.DataFrame(diction)
    df.to_csv(os.path.join(absolute_path, 'logs/losses.csv'))
    

cfg_path = os.path.join(absolute_path, 'Resources/train_FNO.yml')
cfg = OmegaConf.load(cfg_path) 
data_path = cfg.paths.data_path
if isinstance(cfg.model.learning_rate, str):
    lr = float(cfg.model.learning_rate)
else:
    lr = cfg.model.learning_rate

batch_size = cfg.model.batch_size
max_epochs = cfg.model.max_epochs
learning_rate = lr
thickness = 3.8
optimizer_fn = torch.optim.AdamW

#####################################################################################################################################
#################################################### Data preperation ###############################################################

d_path = os.path.join(data_path, 'Trainingsdaten_1D_heavy_complex')
o_path = os.path.join(data_path, 'DoE_Therm_stress_1D_heavy_complex.xlsx')
singleMulti_ds = SingleStepMultiTempDataset1D(dataPath=d_path, overviewPath=o_path, thickness=thickness, size=20, num_input=1,
                                              max_channel=False, time_channel=True, start_temp_channel=False, boundary_temp_channel=False, continuation=False)
input_vector_2 = torch.randn(batch_size, singleMulti_ds.input_size[0], singleMulti_ds.input_size[1])
model_name_2 = "Model_2"


d1_path = os.path.join(data_path, 'Trainingsdaten_1D_testing_complex')
o1_path = os.path.join(data_path, 'DoE_Therm_stress_1D_testing_complex.xlsx')
testing_ds1 = SingleStepMultiTempDataset1D(dataPath=d1_path, overviewPath=o1_path, thickness=thickness, size=20, num_input=1,
                                              max_channel=False, time_channel=True, start_temp_channel=False, boundary_temp_channel=False, continuation=False)

d2_path = os.path.join(data_path, 'Trainingsdaten_1D_testing_1000')
o2_path = os.path.join(data_path, 'DoE_Therm_stress_1D_testing_1000.xlsx')
testing_ds2 = SingleStepMultiTempDataset1D(dataPath=d2_path, overviewPath=o2_path, thickness=thickness, size=1000, num_input=1,
                                              max_channel=False, time_channel=True, start_temp_channel=False, boundary_temp_channel=False, continuation=False)


#####################################################################################################################################


data = singleMulti_ds
testing_data1 = testing_ds1
testing_data2 = testing_ds2
model_description = model_name_2
input_vector = input_vector_2
write = SummaryWriter("runs/Overview/FNO_" + str(datetime.datetime.now().strftime("%y%m%d_%H%M")))

temperature_size = data.input_size[1]
start_channels = data.input_size[0]
print("Start Channels: ", start_channels)
n_splits = 2 # 5
test_size = int(0.1 * data.input_count)
print("Test Size: ", test_size)
print("Input Size: ", data.input_size)
num_workers = 8


testing_indices = np.linspace(0, testing_data1.__len__() - 1, testing_data1.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data1, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader1 = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data1.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)

testing_indices = np.linspace(0, testing_data2.__len__() - 1, testing_data2.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data2, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader2 = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data2.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)#
testing_rel_losses2 = []

#####################################################################################################################################
######################################################## Training ###################################################################
writer = write
train_counter = 0
val_counter = 0

torch.manual_seed(42)

data_indices = np.linspace(0, data.__len__() - 1, data.__len__(), dtype=np.int16)
train_indices, val_indices = train_test_split(data_indices, test_size=test_size, shuffle=False)
time_series_train_loss = []
time_series_valid_loss = []

train_subset = torch.utils.data.Subset(data, train_indices)
val_subset = torch.utils.data.Subset(data, val_indices)
train_sampler = torch.utils.data.SequentialSampler(train_subset.indices)
val_sampler = torch.utils.data.SequentialSampler(val_subset.indices)
train_dataloader = torch.utils.data.DataLoader(train_subset, sampler=train_sampler, collate_fn=data.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_subset, sampler=val_sampler, collate_fn=data.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)

train_abs_loss = []
val_abs_loss = []
val_batch_loss = []
train_batch_loss = []
train_max_diff = []
train_min_diff = []
val_max_diff = []
val_min_diff = []
train_loss = []
val_loss = []
execution_time = []
train_counter = 0
val_counter = 0
testing1_counter = 0
testing2_counter = 0


stopping_path = "logs/EarlyStopping_overview.pth"
stop_path = os.path.join(absolute_path, stopping_path)
earlyStopping = EarlyStopping(start_epoch=450, path=stop_path, patience=100)

model = FNO(start_channels=start_channels, out_channels=1, fourier_modes=16, width=64).to(device)
pde_relative = PDERelative(temperature_input=1).to(device)

epochs = max_epochs
num_batches = len(train_dataloader)
iterations = epochs * num_batches

optimizer = torch.optim.AdamW(list(model.parameters()) + list(pde_relative.parameters()), lr=learning_rate, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
loss_fn = RelativeLoss() 

t1 = default_timer()
torch.autograd.set_detect_anomaly(True)


for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    epoch_abs_loss = 0.0
    batch_train_loss = 0.0
    max_diff = 0
    min_diff = 10000000     
    with tqdm(train_dataloader, desc="Processing", leave=False) as pbar:
        for step, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            x_list, y = x, y
            x = x_list[0]
            
            # Train the model
            x, y = x.to(device, dtype=torch.float32).requires_grad_(True), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            y_pred = y_pred[:, :, :y.shape[2]]
            current_time_list = [data.time_list[step + i] for i in range(y_pred.shape[0])]
            current_time_list = torch.tensor(current_time_list).to(device)
            
            loss, loss_mean = loss_fn(y_pred, y)  
            loss_pde = pde_relative(x=x, prediction=y_pred, target=y, time=current_time_list)
            
            loss = pde_relative.w * loss + loss_pde
             
            pbar.set_description(f"Training epoch {epoch+1}/{epochs} --- Batch Test Loss: {loss.item()} --- ")
            diff = torch.abs(y_pred - y)
            diff = diff.cpu().detach()
            if torch.max(diff) >= max_diff:
                max_diff = torch.max(diff).item()
            if torch.min(diff) <= min_diff:
                min_diff = torch.min(diff).item()
            loss.backward()
            optimizer.step()
            scheduler.step()    
            train_counter += 1
            epoch_train_loss += loss_mean.item()
            epoch_abs_loss += torch.mean(diff).item()
            batch_train_loss += loss.item() * batch_size
    train_loss.append(epoch_train_loss / len(train_dataloader))
    train_abs_loss.append(epoch_abs_loss / len(train_dataloader))
    train_batch_loss.append(batch_train_loss / (len(train_dataloader) * batch_size))
    writer.add_scalar(f"Series Training epoch loss", epoch_train_loss / len(train_dataloader), epoch)
    writer.add_scalar(f"Series Training absolut loss", epoch_abs_loss / len(train_dataloader), epoch)
    train_max_diff.append(max_diff)
    train_min_diff.append(min_diff)
    
    #Validation step
    epoch_val_loss = 0
    epoch_abs_loss = 0
    batch_val_loss = 0
    max_diff = 0
    min_diff = 10000000
    model.eval()
    with torch.no_grad():
        with tqdm(val_dataloader, desc="Processing", leave=False) as pbar:
            for x, y in pbar:
                x_list, y = x, y
                x = x_list[0]
    
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
                y_pred = model(x)
                y_pred = y_pred[:, :, :y.shape[2]]
                loss, loss_mean = loss_fn(y_pred, y) 
                
                pbar.set_description(f"Validation epoch {epoch+1}/{epochs} --- Validation Batch Loss: {loss.item()} --- ")
                diff = torch.abs(y_pred - y)
                diff = diff.cpu().detach()
                if torch.max(diff)>= max_diff:
                    max_diff = torch.max(diff).item()
                if torch.min(diff)<= min_diff:
                    min_diff = torch.min(diff).item()
                val_counter += 1
                epoch_val_loss += loss_mean.item()
                epoch_abs_loss += torch.mean(diff).item()
                batch_val_loss += loss.item() * batch_size
        val_loss.append(epoch_val_loss / len(val_dataloader))
        val_abs_loss.append(epoch_abs_loss / len(val_dataloader))
        val_batch_loss.append(batch_val_loss / (len(val_dataloader) * batch_size))
        writer.add_scalar(f"Series Validation epoch loss", epoch_val_loss / len(val_dataloader), epoch)
        writer.add_scalar(f"Series Validation absolut loss", epoch_abs_loss / len(val_dataloader), epoch)
        val_max_diff.append(max_diff)
        val_min_diff.append(min_diff)
        
    epoch_val_loss = 0
    epoch_abs_loss = 0
    batch_val_loss = 0
    max_diff = 0
    min_diff = 10000000
    model.eval()
    with torch.no_grad():
        with tqdm(testing_dataloader1, desc="Processing", leave=False) as pbar:
            for x, y in pbar:
                x_list, y = x, y
                x = x_list[0]
        
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
                y_pred = model(x)
                y_pred = y_pred[:, :, :y.shape[2]]
                loss, loss_mean = loss_fn(y_pred, y) 
            
                pbar.set_description(f"Validation epoch {epoch+1}/{epochs} --- Validation Batch Loss: {loss.item()} --- ")
                diff = torch.abs(y_pred - y)
                diff = diff.cpu().detach()
                if torch.max(diff)>= max_diff:
                    max_diff = torch.max(diff).item()
                if torch.min(diff)<= min_diff:
                    min_diff = torch.min(diff).item()
                testing1_counter += 1
                epoch_val_loss += loss_mean.item()
                epoch_abs_loss += torch.mean(diff).item()
                batch_val_loss += loss.item() * batch_size
        writer.add_scalar(f"Series Validation epoch loss same", epoch_val_loss / len(testing_dataloader1), epoch)
        writer.add_scalar(f"Series Validation absolut loss same", epoch_abs_loss / len(testing_dataloader1), epoch)
    earlyStopping(epoch=epoch, loss=epoch_abs_loss / len(testing_dataloader1), model=model)
    if earlyStopping.stopping:
        break
time_series_train_loss.append(train_loss)
time_series_valid_loss.append(val_loss)
t2 = default_timer()
execution_time.append(t2 - t1)
print(f"Execution Time: {t2 - t1}s")    
    

for epoch in range(len(time_series_train_loss[0])):
    train_dict = {}
    valid_dict = {}
    for series, losses in enumerate(time_series_train_loss):
        if epoch < len(losses):
            train_dict[f'Series_{series}'] = losses[epoch]
    for series, losses in enumerate(time_series_valid_loss):
        if epoch < len(losses):
            valid_dict[f'Series_{series}'] = losses[epoch]
    write.add_scalars("Overall Training Loss", train_dict, epoch)
    write.add_scalars("Overall Validation Loss", valid_dict, epoch)

# Visualization for one training example y_pred = (batch, 1, x)
for time, (pred, target) in enumerate(zip(y_pred[0].flatten(), y[0].flatten())):
    writer.add_scalars("Validation: Prediction vs. Target", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(y_pred[-1].flatten(), y[-1].flatten())):
    writer.add_scalars("Validation: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)



############################## Load Best Model ####################################
if earlyStopping.stopping:
    model.load_state_dict(torch.load(stop_path))

###################################################################################


############################# Testing same size ###################################

testing_indices = np.linspace(0, testing_data1.__len__() - 1, testing_data1.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data1, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data1.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)

testing_rel_losses1 = []
testing_abs_losses1 = []
testing_pred1 = []
testing_y1 = []
diff_losses1 = []
testing_val_loss1 = 0
testing_abs_loss1 = 0
val_counter1 = 0

model.eval()

with torch.no_grad():
    with tqdm(testing_dataloader, desc="Processing", leave=False) as pbar:
        for x, y in pbar:
            x_list, y = x, y
        
            x = x_list[0]

            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            y_pred = y_pred[:, :, :y.shape[2]]
            
            testing_pred1.append(y_pred)
            testing_y1.append(y)
            
            loss, loss_mean = loss_fn(y_pred, y) 
            pbar.set_description(f"Testing Batch Loss: {loss_mean.item()} --- ")
            diff = torch.abs(y_pred - y)
            diff = diff.cpu().detach()
            diff_losses1.append(diff)
            val_counter1 += 1
            testing_val_loss1 += loss_mean.item()
            testing_abs_loss1 += torch.mean(diff).item()
    testing_rel_losses1.append(testing_val_loss1 / len(testing_dataloader))
    testing_abs_losses1.append(testing_abs_loss1 / len(testing_dataloader))


# Visualization for one training example y_pred = (batch, 1, x)
for time, (pred, target) in enumerate(zip(testing_pred1[3][0][0].flatten(), testing_y1[3][0][0].flatten())):
    writer.add_scalars("Testing 1 same size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[7][0][0].flatten(), testing_y1[7][0][0].flatten())):
    writer.add_scalars("Testing 2 same size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[-5][0][0].flatten(), testing_y1[-5][0][0].flatten())):
    writer.add_scalars("Testing 3 same size: Prediction vs. Target", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[-1][0][0].flatten(), testing_y1[-1][0][0].flatten())):
    writer.add_scalars("Testing 4 same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)

for time, (pred, target) in enumerate(zip(testing_pred1[0][0][0].flatten(), testing_y1[0][0][0].flatten())):
    writer.add_scalars("Testing 5 same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[0][4][0].flatten(), testing_y1[0][4][0].flatten())):
    writer.add_scalars("Testing 6 same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[1][0][0].flatten(), testing_y1[1][0][0].flatten())):
    writer.add_scalars("Testing 7 same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
######################################################################################
    

################################### Testing different size ########################################

testing_indices = np.linspace(0, testing_data2.__len__() - 1, testing_data2.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data2, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data2.collate_function, batch_size=batch_size, num_workers=num_workers, shuffle=False)#
testing_rel_losses2 = []
testing_abs_losses2 = []
testing_pred2 = []
testing_y2 = []
diff_losses2 = []
testing_val_loss2 = 0
testing_abs_loss2 = 0
val_counter2 = 0#
model.eval()

with torch.no_grad():
    with tqdm(testing_dataloader, desc="Processing", leave=False) as pbar:
        for x, y in pbar:
            x_list, y = x, y
        
            x = x_list[0]
            
            # Train the model
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            y_pred = y_pred[:, :, :y.shape[2]]
            
            testing_pred2.append(y_pred)
            testing_y2.append(y)
            
            loss, loss_mean = loss_fn(y_pred, y) 
            pbar.set_description(f"Testing Batch Loss: {loss_mean.item()} --- ")
            diff = torch.abs(y_pred - y)
            diff = diff.cpu().detach()
            diff_losses2.append(diff)
            val_counter2 += 1
            testing_val_loss2 += loss_mean.item()
            testing_abs_loss2 += torch.mean(diff).item()
    testing_rel_losses2.append(testing_val_loss2 / len(testing_dataloader))
    testing_abs_losses2.append(testing_abs_loss2 / len(testing_dataloader))

# Visualization for one training example y_pred = (batch, 1, x)
for time, (pred, target) in enumerate(zip(testing_pred2[3][0][0].flatten(), testing_y2[3][0][0].flatten())):
    writer.add_scalars("Testing 1 different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[7][0][0].flatten(), testing_y2[7][0][0].flatten())):
    writer.add_scalars("Testing 2 different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[-5][0][0].flatten(), testing_y2[-5][0][0].flatten())):
    writer.add_scalars("Testing 3 different size: Prediction vs. Target", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[-1][0][0].flatten(), testing_y2[-1][0][0].flatten())):
    writer.add_scalars("Testing 4 different size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[0][0][0].flatten(), testing_y2[0][0][0].flatten())):
    writer.add_scalars("Testing 5 different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[0][4][0].flatten(), testing_y2[0][4][0].flatten())):
    writer.add_scalars("Testing 6 different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)

for time, (pred, target) in enumerate(zip(testing_pred2[1][0][0].flatten(), testing_y2[1][0][0].flatten())):
    writer.add_scalars("Testing 7 different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)   

#####################################################################################################################################


#####################################################################################################################################
######################################################## Evaluation #################################################################

print(f"Testing relative Error same size: {(testing_rel_losses1[0] * 100):.4f}%")
print(f"Testing absolute Error same size: {(testing_abs_losses1[0]):.4f}")

data_points = testing_data1.input_size[1]
start_index = int(data_points * 0.1) // 2 - (data_points // 2)
end_index = start_index + int(data_points * 0.1)

middle_loss = []

for batch_loss in diff_losses1:
    sample_mean = 0
    for sample in batch_loss:
        sample_mean += np.mean([torch.mean(s[start_index:end_index]).numpy() for s in sample])
    middle_loss.append(sample_mean / len(sample))
        
print(f"Middle area absolute Error same size: {(np.mean(middle_loss)):.4f}")


data_points = testing_data1.input_size[1]
start_index = int(data_points * 0.05) 

boundary_loss = []

for batch_loss in diff_losses1:
    sample_mean = 0
    for sample in batch_loss:
        sample_mean += np.mean([torch.mean(s[0:start_index]).numpy() + torch.mean(s[-start_index:-1]).numpy()  for s in sample])
    boundary_loss.append(sample_mean / batch_size)
        
print(f"Boundary area absolute Error same size: {(np.mean(boundary_loss)):.4f}")




print(f"Testing relative Error different size: {(testing_rel_losses2[0] * 100):.4f}%")
print(f"Testing absolute Error different size: {(testing_abs_losses2[0]):.4f}")

data_points = testing_data2.input_size[1]
start_index = int(data_points * 0.1) // 2 - (data_points // 2)
end_index = start_index + int(data_points * 0.1)

middle_loss = []

for batch_loss in diff_losses2:
    sample_mean = 0
    for sample in batch_loss:
        sample_mean += np.mean([torch.mean(s[start_index:end_index]).numpy() for s in sample])
    middle_loss.append(sample_mean / len(sample))
        
print(f"Middle area absolute Error different size: {(np.mean(middle_loss)):.4f}")


data_points = testing_data2.input_size[1]
start_index = int(data_points * 0.05) 

boundary_loss = []

for batch_loss in diff_losses2:
    sample_mean = 0
    for sample in batch_loss:
        sample_mean += np.mean([torch.mean(s[0:start_index]).numpy() + torch.mean(s[-start_index:-1]).numpy()  for s in sample])
    boundary_loss.append(sample_mean / batch_size)
        
print(f"Boundary area absolute Error different size: {(np.mean(boundary_loss)):.4f}")

#####################################################################################################################################


############################# Testing same size error propagation ###################################

testing_indices = np.linspace(0, testing_data1.__len__() - 1, testing_data1.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data1, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data1.collate_function, num_workers=num_workers, shuffle=False)

testing_rel_losses1 = []
testing_abs_losses1 = []
testing_pred1 = []
testing_y1 = []
diff_losses1 = []
testing_val_loss1 = 0
testing_abs_loss1 = 0
val_counter1 = 0
counter = 0

model.eval()

with torch.no_grad():
    with tqdm(testing_dataloader, desc="Processing", leave=False) as pbar:
        for _x, y in pbar:
            _x, y = _x[0], y
            
            # Update time channel
            if _x[0][-1][0] == torch.tensor(0.0, dtype=torch.float32):
                x = _x
            else:
                x[:, -1:, :] = _x[:, -1:, :]  
            
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            saved = y_pred
            y_pred = y_pred[:, :, :y.shape[2]]
            
            testing_pred1.append(y_pred)
            testing_y1.append(y)
            
            loss, loss_mean = loss_fn(y_pred, y) 
            pbar.set_description(f"Testing Batch Loss: {loss_mean.item()} --- ")
            diff = torch.abs(y_pred - y)
            diff = diff.cpu().detach()
            diff_losses1.append(diff)
            val_counter1 += 1
            testing_val_loss1 += loss_mean.item()
            testing_abs_loss1 += torch.mean(diff).item()
            writer.add_scalar(f"Absolute Propagation loss same size", torch.mean(diff).item(), counter)
            
            last_relative_loss1 = loss_mean
            last_absolut_loss1 = torch.mean(diff)
            
            counter += 1
            
            # Output next input
            x = _x
            x[0][1] = saved   
            
    testing_rel_losses1.append(testing_val_loss1 / len(testing_dataloader))
    testing_abs_losses1.append(testing_abs_loss1 / len(testing_dataloader))
    

for time, (pred, target) in enumerate(zip(testing_pred1[0][0][0].flatten(), testing_y1[0][0][0].flatten())):
    writer.add_scalars("Propagation 0s same size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[1][0][0].flatten(), testing_y1[1][0][0].flatten())):
    writer.add_scalars("Propagation 4s same size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)

# Visualization for one training example y_pred = (batch, 1, x)
for time, (pred, target) in enumerate(zip(testing_pred1[2][0][0].flatten(), testing_y1[2][0][0].flatten())):
    writer.add_scalars("Propagation 8s same size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[5][0][0].flatten(), testing_y1[5][0][0].flatten())):
    writer.add_scalars("Propagation 20s same size: Prediction vs. Target", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[10][0][0].flatten(), testing_y1[10][0][0].flatten())):
    writer.add_scalars("Propagation 40s same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred1[20][0][0].flatten(), testing_y1[20][0][0].flatten())):
    writer.add_scalars("Propagation 80s same size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)



################################### Testing different size error propagation ########################################

testing_indices = np.linspace(0, testing_data2.__len__() - 1, testing_data2.__len__(), dtype=np.int16)
testing_subset = torch.utils.data.Subset(testing_data2, testing_indices)
testing_sampler = torch.utils.data.SequentialSampler(testing_subset.indices)
testing_dataloader = torch.utils.data.DataLoader(testing_subset, sampler=testing_sampler, collate_fn=testing_data2.collate_function, num_workers=num_workers, shuffle=False)
testing_rel_losses2 = []
testing_abs_losses2 = []
testing_pred2 = []
testing_y2 = []
diff_losses2 = []
testing_val_loss2 = 0
testing_abs_loss2 = 0
val_counter2 = 0
counter = 0
model.eval()

with torch.no_grad():
    with tqdm(testing_dataloader, desc="Processing", leave=False) as pbar:
        for _x, y in pbar:
            _x, y = _x[0], y
            
            # Update time channel
            if _x[0][-1][0] == torch.tensor(0.0, dtype=torch.float32):
                x = _x
            else:
                x[:, -1:, :] = _x[:, -1:, :] 
            
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = model(x)
            saved = y_pred
            y_pred = y_pred[:, :, :y.shape[2]]
            
            testing_pred2.append(y_pred)
            testing_y2.append(y)
            
            loss, loss_mean = loss_fn(y_pred, y) 
            pbar.set_description(f"Testing Batch Loss: {loss_mean.item()} --- ")
            diff = torch.abs(y_pred - y)
            diff = diff.cpu().detach()
            diff_losses2.append(diff)
            val_counter2 += 1
            testing_val_loss2 += loss_mean.item()
            testing_abs_loss2 += torch.mean(diff).item()
            writer.add_scalar(f"Absolute Propagation loss different size", torch.mean(diff).item(), counter)
            
            last_relative_loss2 = loss_mean
            last_absolut_loss2 = torch.mean(diff)
            
            counter += 1
            
            # Output next input
            x = _x
            x[0][1] = saved
            
    testing_rel_losses2.append(testing_val_loss2 / len(testing_dataloader))
    testing_abs_losses2.append(testing_abs_loss2 / len(testing_dataloader))

for time, (pred, target) in enumerate(zip(testing_pred2[0][0][0].flatten(), testing_y2[0][0][0].flatten())):
    writer.add_scalars("Propagation 0s different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[1][0][0].flatten(), testing_y2[1][0][0].flatten())):
    writer.add_scalars("Propagation 4s different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
# Visualization for one training example y_pred = (batch, 1, x)
for time, (pred, target) in enumerate(zip(testing_pred2[2][0][0].flatten(), testing_y2[2][0][0].flatten())):
    writer.add_scalars("Propagation 8s different size: Prediction first vs. Target first", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[5][0][0].flatten(), testing_y2[5][0][0].flatten())):
    writer.add_scalars("Propagation 20s different size: Prediction vs. Target", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[10][0][0].flatten(), testing_y2[10][0][0].flatten())):
    writer.add_scalars("Propagation 40s different size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)
    
for time, (pred, target) in enumerate(zip(testing_pred2[20][0][0].flatten(), testing_y2[20][0][0].flatten())):
    writer.add_scalars("Propagation 80s different size: Prediction last vs. Target last", {"Prediction": pred.item(), "Target": target.item()}, time)


writer.close()


print("Error Propagation:")
print(f"Testing relative Error same size: {(testing_rel_losses1[0] * 100):.4f}%")
print(f"Testing absolute Error same size: {(testing_abs_losses1[0]):.4f}")
print(f"Testing relative Error last picture same size: {(last_relative_loss1 * 100):.4f}%")
print(f"Testing absolute Error last picture same size: {(last_absolut_loss1):.4f}")

print(f"Testing relative Error different size: {(testing_rel_losses2[0] * 100):.4f}%")
print(f"Testing absolute Error different size: {(testing_abs_losses2[0]):.4f}")
print(f"Testing relative Error last picture different size: {(last_relative_loss2 * 100):.4f}%")
print(f"Testing absolute Error last picture different size: {(last_absolut_loss2):.4f}")

