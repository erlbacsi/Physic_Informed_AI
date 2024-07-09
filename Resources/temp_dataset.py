import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import re
from torchvision import transforms


def extract_number(file):
    numbers = re.search(r'\d+.\d+$', file)
    number = float(numbers.group()) if numbers else -1
    return number


def extract_numbers_as_list(files):
    numbers = []
    for s in files:
        number = re.search(r'\d+.\d+$', s)
        number = float(number.group()) if number else -1
        numbers.append(number)
    numbers.sort()
    return numbers


class MultiStepMultiTempDataset1D(Dataset):
    """
    Returns a batch of x, y tuples of the shape
    x: [batch_size, 71]
    y: [batch_size, 1, 21, 2001]
    """

    # difference = 0 bedeutet, dass die der Abstand t = 1 ist standardmäßig -> t_i auf t_i + 1
    def __init__(self, dataPath, overviewPath, thickness, num_input, size, difference, max_channel=False,
                 time_channel=False, start_temp_channel=False, boundary_temp_channel=False, continuation=False) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_output = num_input
        self.zone_numbers = 0
        self.difference = difference
        self.time_list = []
        # self.time_per_step = 400
        # self.solutionPath = os.path.join(dataPath, "Trainingsdaten_1D_testing")
        self.solutionPath = dataPath
        self.data = os.listdir(self.solutionPath)
        # First peek into the data
        # data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress_1D_testing.xlsx'))
        data = pd.ExcelFile(overviewPath)

        # Load the zone values and name/thickness vals
        self.df = data.parse(sheet_name=str(thickness) + 'mm', skiprows=0)
        cols1 = list(self.df.columns)
        names = list(self.df.iloc[0, 1:])
        self.col_names = ["Attribute"]
        self.col_names.extend(names)
        self.zone_end = None
        zones, conditions, material = self.getDataFromFrame(self.df)

        # create a numpy array with all the zones as input for the neural net
        tmp_zones = []
        for zone in zones:
            z = zone.iloc[:, 1:].to_numpy(dtype=float)
            tmp_zones.append(z)
        temp_zones = np.asarray(tmp_zones)

        # Create numpy arrays from the other conditions
        condition_input = conditions.iloc[:, 1:].to_numpy()
        material_input = material.iloc[:, 1:].to_numpy()
        material_input[0] = material_embedding(material_input[0])
        # thick = self.df.iloc[1, 1:].to_numpy()
        # print('thickness array:',thickness)
        # timesteps in s
        timesteps = self.df.iloc[35, 1:].to_numpy()

        # Add all the inputs together for the different simulation runs
        self.inputs_lst = []
        self.outputs_lst = []
        # dir_base_name = "TD_10_{}"
        dir_base_name = "" + str(size) + "_{}"
        for i in range(temp_zones.shape[-1]):
            dir_name = dir_base_name.format(i + 1)
            file_names = []
            file_numbers = []
            # extract file information
            for file in os.listdir(os.path.join(self.solutionPath, dir_name)):
                file_names.append(file)
            file_numbers = extract_numbers_as_list(file_names)
            file_names = sorted(file_names, key=extract_number)
            for j in range(len(file_numbers) - (self.num_input + self.difference)):
                # Temperature input
                temperature_in = []
                time_input = []
                for k in range(self.num_input):
                    file_name = file_names[j + k]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    if k == 0:
                        coord_in = data.iloc[:, :1].to_numpy()
                        coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        if continuation is True:
                            pot = np.ceil(np.log2(coord_in.shape[1]))
                            size = 2 ** (pot + 1)
                            pad_size = int(size - coord_in.shape[1])
                            coord_in = np.pad(coord_in, ((0, 0), (0, pad_size)), "constant")
                        temperature_in.append(coord_in)
                    temp_in = data.iloc[:, -1].to_numpy()
                    # Reshape soll Dimension des realen Teils wiederspiegeln -> im eindimenionsalen eine Dimension 1
                    temp_in = temp_in.reshape((1, temp_in.shape[0]))
                    current_time = np.array([file_numbers[j+k] / 100])
                    # Continuation Temperature padding
                    if continuation is True:
                        pad_size = int(size - temp_in.shape[1])
                        temp_in = np.pad(temp_in, ((0, 0), (0, pad_size)), "constant")
                    temperature_in.append(temp_in)
                    time_in = np.ones_like(temp_in) * current_time
                    time_input.append(current_time.item())
                    if time_channel:
                        temperature_in.append(time_in)
                    
                temperature_in = np.concatenate(temperature_in, axis=0)
                self.inputs_lst.append([dir_name, torch.from_numpy(temperature_in).to(dtype=torch.float32)])
                self.time_list.append(time_input)

                # Boundarys
                size = temperature_in.shape[1]
                last_index = len(self.inputs_lst) - 1
                temp_zone = temp_zones[:, :, i].flatten()
                condition = condition_input[:, i].flatten()
                thickness_input = np.expand_dims(thickness, 0)
                # timestep_input = np.expand_dims(timesteps[i], 0)
                material_input_exp = material_input[:, i].flatten()  # np.squeeze(material_input, axis=-1)
                # The number in the data shows the steps -> one step = 0.01s
                current_time = np.array([file_numbers[j] / 100])
                boundary = [temp_zone, condition, material_input_exp, thickness_input, current_time]
                # boundary = [temp_zone, condition, material_input_exp, thickness_input]

                if time_channel is True:
                    # time = torch.tensor(boundary[-1])
                    # time = time.expand(1, size)
                    # self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], time), dim=0)
                    boundary = boundary[0:-1]
                if start_temp_channel is True:
                    start_temperature = torch.tensor(condition[0])
                    start_temperature = start_temperature.expand(1, size)
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], start_temperature), dim=0)
                    boundary[1] = condition[1:]
                if max_channel is True:
                    conditions = condition
                    for con in conditions:
                        c = torch.tensor(con)
                        c = c.expand(1, size)
                        self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], c), dim=0)
                    del boundary[1]
                if boundary_temp_channel is True:
                    boundary_temps = len(boundary[0])
                    temps = int(boundary_temps / self.zone_numbers)
                    data_size = self.inputs_lst[0][1].shape[1]
                    start = 0
                    end = 0
                    if j <= 24:
                        start = 0
                        end = temps
                    elif 24 < j <= 51:
                        start = temps
                        end = temps * 2
                    elif 51 < j <= 66:
                        start = temps * 2
                        end = temps * 3
                    elif 66 < j <= 73:
                        start = temps * 3
                        end = temps * 4
                    elif 73 < j <= 74:
                        start = temps * 4
                        end = temps * 5
                    elif 74 < j <= 80:
                        start = temps * 5
                        end = temps * 6
                    elif 80 < j <= 84:
                        start = temps * 6
                        end = temps * 7
                    elif 84 < j <= 86:
                        start = temps * 7
                        end = temps * 8
                    left, right, air = boundary[0][start:end]
                    halfway = data_size // 2
                    first_half = np.linspace(left, air, halfway, endpoint=False)
                    second_half = np.linspace(air, right, data_size - halfway)
                    boundary_temp = torch.tensor(np.concatenate((first_half, second_half))).to(dtype=torch.float32)
                    boundary_temp = boundary_temp.reshape((1, boundary_temp.shape[0]))
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], boundary_temp),
                                                               dim=0)
                    del boundary[0]
                # Temperature and Boundary combined
                boundary = np.concatenate(boundary).astype(float).reshape(1, -1)
                self.inputs_lst[last_index].append(torch.from_numpy(boundary).to(dtype=torch.float32))
                #self.inputs_lst.append((dir_name, torch.tensor(inp), torch.tensor(np.array(temperature_in))))

                # Temperature output
                temperature_out = []
                for y in range(j + difference + 1, j + self.num_output + difference + 1):
                    file_name = file_names[y]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    temp_out = data.iloc[:, -1].to_numpy()
                    # Berechne Anzahl Zeilen und Spalten am besten
                    temp_out = temp_out.reshape((1, temp_out.shape[0]))
                    temperature_out.append(temp_out)
                temperature_out = np.concatenate(temperature_out)
                self.outputs_lst.append((dir_name, torch.tensor(np.array(temperature_out), dtype=torch.float32)))

        self.input_size = self.inputs_lst[0][1].shape
        self.input_count = len(self.inputs_lst)
        self.boundary_size = self.inputs_lst[0][2].shape
        self.output_size = self.outputs_lst[0][1].shape
        self.output_count = len(self.outputs_lst)

    def getDataFromFrame(self, df: pd.DataFrame):
        # Zones
        zones = []
        i = 2
        # count zones
        self.zone_numbers = 0
        while df.iloc[i, 0].startswith("zone"):
            zones.append(df.iloc[i + 1:i + 4, :].dropna(axis=1))
            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
            i += 4
            self.zone_numbers += 1
        self.zone_end = i - self.zone_numbers
        # Conditional Info
        # df.iloc[i:i+5]
        conditions = df.iloc[i:i+12, :]
        conditions.columns = self.col_names[:]
        i += 12
        # Material info (Annahme: 1 Material)
        material = df.iloc[i:i + 9, :].dropna(axis=0)
        material.columns = self.col_names[:]
        return zones, conditions, material

    def getInputParamCount(self):
        return self.input_size

    def collate_function(self, batch):
        in_d, out_d = zip(*batch)
        _, in_tensor, in_bound = zip(*in_d)
        _, out_tensor = zip(*out_d)
        in_temp_tensors = torch.stack(in_tensor)
        in_bound_tensors = torch.stack(in_bound)
        out_tensors = torch.stack(out_tensor)
        return (in_temp_tensors, in_bound_tensors), out_tensors

    def __len__(self):
        return self.input_count

    def __getitem__(self, index):
        # Read the temperature solutions
        params = self.inputs_lst[index]
        solution = self.outputs_lst[index]
        return params, solution
    
class SingleStepMultiTempDataset1D(Dataset):
    """
    Returns a batch of x, y tuples of the shape
    x: [batch_size, 71]
    y: [batch_size, 1, 21, 2001]
    """

    def __init__(self, dataPath, overviewPath, thickness, size, num_input, max_channel=False,
                 time_channel=False, start_temp_channel=False, boundary_temp_channel=False, continuation=False, start=0, switch=False) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_output = 1
        self.zone_numbers = 0
        self.time_list = []
        self.left_pad = 0
        # self.time_per_step = 400
        # self.solutionPath = os.path.join(dataPath, "Trainingsdaten_1D_testing")
        self.solutionPath = dataPath
        self.data = os.listdir(self.solutionPath)
        # First peek into the data
        # data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress_1D_testing.xlsx'))
        data = pd.ExcelFile(overviewPath)

        # Load the zone values and name/thickness vals
        self.df = data.parse(sheet_name=str(thickness) + 'mm', skiprows=0)
        cols1 = list(self.df.columns)
        names = list(self.df.iloc[0, 1:])
        self.col_names = ["Attribute"]
        self.col_names.extend(names)
        self.zone_end = None
        zones, conditions, material = self.getDataFromFrame(self.df)

        # create a numpy array with all the zones as input for the neural net
        tmp_zones = []
        for zone in zones:
            z = zone.iloc[:, 1:].to_numpy(dtype=float)
            tmp_zones.append(z)
        temp_zones = np.asarray(tmp_zones)

        # Create numpy arrays from the other conditions
        condition_input = conditions.iloc[:, 1:].to_numpy()
        material_input = material.iloc[:, 1:].to_numpy()
        material_input[0] = material_embedding(material_input[0])
        # thick = self.df.iloc[1, 1:].to_numpy()
        # print('thickness array:',thickness)
        # timesteps in s
        timesteps = self.df.iloc[35, 1:].to_numpy()

        # Add all the inputs together for the different simulation runs
        self.inputs_lst = []
        self.outputs_lst = []
        # dir_base_name = "TD_10_{}"
        dir_base_name = "" + str(size) + "_{}"

        for i in range(temp_zones.shape[-1]):
            dir_name = dir_base_name.format(i + 1)
            file_names = []
            file_numbers = []
            # extract file information
            for file in os.listdir(os.path.join(self.solutionPath, dir_name)):
                if file == ".ipynb_checkpoints":
                    continue
                file_names.append(file)
            file_numbers = extract_numbers_as_list(file_names)
            file_names = sorted(file_names, key=extract_number)
            if switch:
                start = i % 3
                if start == 2:
                    start = 1
            for j in range(start, len(file_numbers) - self.num_input):
                # Temperature input
                temperature_in = []
                time_input = []
                for k in range(self.num_input):
                    file_name = file_names[j + k]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    # Einmalig zu jedem Trainingsbeispiel Koordinaten mit anlegen als Dimension
                    if k == 0:
                        coord_in = data.iloc[:, :1].to_numpy()
                        coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        # Continuation Coordinates
                        if continuation is True:
                            # step_size = coord_in[0][1] - coord_in[0][0]
                            pot = np.ceil(np.log2(coord_in.shape[1]))
                            size = 2 ** (pot + 1)
                            pad_size = int(size - coord_in.shape[1])
                            
                            # left_pad = pad_size // 2
                            # right_pad = pad_size - left_pad
                            # self.left_pad = left_pad
                            
                            coord_in = np.pad(coord_in, ((0, 0), (0, pad_size)), "constant")
                            # coord_in = np.array([x * step_size for x in range(int(size))])
                            # coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        temperature_in.append(coord_in)
                    temp_in = data.iloc[:, -1].to_numpy()
                    # Reshape soll Dimension des realen Teils wiederspiegeln -> im eindimenionsalen eine Dimension 1
                    temp_in = temp_in.reshape((1, temp_in.shape[0]))
                    current_time = np.array([file_numbers[j+k] / 100])  
                    # Continuation Temperature padding
                    if continuation is True:
                        pad_size = int(size - temp_in.shape[1])
                        
                        # left_pad = pad_size // 2
                        # right_pad = pad_size - left_pad
                        
                        temp_in = np.pad(temp_in, ((0, 0), (0, pad_size)), "constant")
                    temperature_in.append(temp_in)
                    time_in = np.ones_like(temp_in) * current_time
                    time_input.append(current_time.item())
                    if time_channel:
                       temperature_in.append(time_in)
                temperature_in = np.concatenate(temperature_in, axis=0)
                self.inputs_lst.append([dir_name, torch.from_numpy(temperature_in).to(dtype=torch.float32)])
                self.time_list.append(time_input)

                # Boundarys
                size = temperature_in.shape[1]
                last_index = len(self.inputs_lst) - 1
                temp_zone = temp_zones[:, :, i].flatten()
                condition = condition_input[:, i].flatten()
                thickness_input = np.expand_dims(thickness, 0)
                # timestep_input = np.expand_dims(timesteps[i], 0)
                material_input_exp = material_input[:, i].flatten()  # np.squeeze(material_input, axis=-1)
                # The number in the data shows the steps -> one step = 0.01s
                current_time = np.array([file_numbers[j] / 100])
                boundary = [temp_zone, condition, material_input_exp, thickness_input, current_time]

                if time_channel is True:
                    # time = torch.tensor(boundary[-1])
                    # time = time.expand(1, size)
                    # self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], time), dim=0)
                    boundary = boundary[0:-1]
                if start_temp_channel is True:
                    start_temperature = torch.tensor(condition[0])
                    start_temperature = start_temperature.expand(1, size)
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], start_temperature),
                                                               dim=0)
                    boundary[1] = condition[1:]
                if max_channel is True:
                    conditions = condition
                    for con in conditions:
                        c = torch.tensor(con)
                        c = c.expand(1, size)
                        self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], c), dim=0)
                    del boundary[1]
                # boundary temperature manuell für jedes Bild
                if boundary_temp_channel is True:
                    boundary_temps = len(boundary[0])
                    temps = int(boundary_temps / self.zone_numbers)
                    data_size = self.inputs_lst[0][1].shape[1]
                    start = 0
                    end = 0
                    if j <= 24:
                        start = 0
                        end = temps
                    elif 24 < j <= 51:
                        start = temps
                        end = temps * 2
                    elif 51 < j <= 66:
                        start = temps * 2
                        end = temps * 3
                    elif 66 < j <= 73:
                        start = temps * 3
                        end = temps * 4
                    elif 73 < j <= 74:
                        start = temps * 4
                        end = temps * 5
                    elif 74 < j <= 80:
                        start = temps * 5
                        end = temps * 6
                    elif 80 < j <= 84:
                        start = temps * 6
                        end = temps * 7
                    elif 84 < j <= 86:
                        start = temps * 7
                        end = temps * 8
                    left, right, air = boundary[0][start:end]
                    halfway = data_size // 2
                    first_half = np.linspace(left, air, halfway, endpoint=False)
                    second_half = np.linspace(air, right, data_size - halfway)
                    boundary_temp = torch.tensor(np.concatenate((first_half, second_half))).to(dtype=torch.float32)
                    boundary_temp = boundary_temp.reshape((1, boundary_temp.shape[0]))  
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], boundary_temp),
                                                               dim=0)
                    del boundary[0]
                # Temperature and Boundary combined
                boundary = np.concatenate(boundary).astype(float).reshape(1, -1)
                self.inputs_lst[last_index].append(torch.from_numpy(boundary).to(dtype=torch.float32))
                # self.inputs_lst.append((dir_name, torch.tensor(inp), torch.tensor(np.array(temperature_in))))

                # Temperature output
                # temperature_out = []
                for y in range(j + k + 1, j + k + self.num_output + 1):
                    file_name = file_names[y]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    temp_out = data.iloc[:, -1].to_numpy()
                    # Berechne Anzahl Zeilen und Spalten am besten
                    temperature_out = temp_out.reshape((1, temp_out.shape[0]))
                    # temperature_out.append(temp_out)
                self.outputs_lst.append((dir_name, torch.tensor(np.array(temperature_out), dtype=torch.float32)))

        self.input_size = self.inputs_lst[0][1].shape
        self.input_count = len(self.inputs_lst)
        self.boundary_size = self.inputs_lst[0][2].shape
        self.output_size = self.outputs_lst[0][1].shape
        self.output_count = len(self.outputs_lst)

    def getDataFromFrame(self, df: pd.DataFrame):
        # Zones
        zones = []
        i = 2
        # count zones
        self.zone_numbers = 0
        while df.iloc[i, 0].startswith("zone"):
            zones.append(df.iloc[i + 1:i + 4, :].dropna(axis=1))
            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
            i += 4
            self.zone_numbers += 1
        self.zone_end = i - self.zone_numbers
        # Conditional Info
        # df.iloc[i:i+5]
        conditions = df.iloc[i:i+12, :]
        conditions.columns = self.col_names[:]
        i += 12
        # Material info (Annahme: 1 Material)
        material = df.iloc[i:i + 9, :].dropna(axis=0)
        material.columns = self.col_names[:]
        return zones, conditions, material

    def getInputParamCount(self):
        return self.input_size

    def collate_function(self, batch):
        in_d, out_d = zip(*batch)
        _, in_tensor, in_bound = zip(*in_d)
        _, out_tensor = zip(*out_d)
        in_temp_tensors = torch.stack(in_tensor)
        in_bound_tensors = torch.stack(in_bound)
        out_tensors = torch.stack(out_tensor)
        return (in_temp_tensors, in_bound_tensors), out_tensors

    def __len__(self):
        return self.input_count

    def __getitem__(self, index):
        # Read the temperature solutions
        params = self.inputs_lst[index]
        solution = self.outputs_lst[index]
        return params, solution


class SingleStepMultiTempDataset1Dtesting(Dataset):
    """
    Returns a batch of x, y tuples of the shape
    x: [batch_size, 71]
    y: [batch_size, 1, 21, 2001]
    """

    def __init__(self, dataPath, overviewPath, thickness, size, num_input, indices, max_channel=False,
                 time_channel=False, start_temp_channel=False, boundary_temp_channel=False, continuation=False) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_output = 1
        self.zone_numbers = 0
        self.time_list = []
        self.indices = indices
        # self.time_per_step = 400
        # self.solutionPath = os.path.join(dataPath, "Trainingsdaten_1D_testing")
        self.solutionPath = dataPath
        self.data = os.listdir(self.solutionPath)
        # First peek into the data
        # data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress_1D_testing.xlsx'))
        data = pd.ExcelFile(overviewPath)

        # Load the zone values and name/thickness vals
        self.df = data.parse(sheet_name=str(thickness) + 'mm', skiprows=0)
        cols1 = list(self.df.columns)
        names = list(self.df.iloc[0, 1:])
        self.col_names = ["Attribute"]
        self.col_names.extend(names)
        self.zone_end = None
        zones, conditions, material = self.getDataFromFrame(self.df)

        # create a numpy array with all the zones as input for the neural net
        tmp_zones = []
        for zone in zones:
            z = zone.iloc[:, 1:].to_numpy(dtype=float)
            tmp_zones.append(z)
        temp_zones = np.asarray(tmp_zones)

        # Create numpy arrays from the other conditions
        condition_input = conditions.iloc[:, 1:].to_numpy()
        material_input = material.iloc[:, 1:].to_numpy()
        material_input[0] = material_embedding(material_input[0])
        # thick = self.df.iloc[1, 1:].to_numpy()
        # print('thickness array:',thickness)
        # timesteps in s
        timesteps = self.df.iloc[35, 1:].to_numpy()

        # Add all the inputs together for the different simulation runs
        self.inputs_lst = []
        self.outputs_lst = []
        # dir_base_name = "TD_10_{}"
        dir_base_name = "" + str(size) + "_{}"
        for i in self.indices:
        #for i in range(temp_zones.shape[-1]):
            dir_name = dir_base_name.format(i + 1)
            file_names = []
            file_numbers = []
            # extract file information
            for file in os.listdir(os.path.join(self.solutionPath, dir_name)):
                file_names.append(file)
            file_numbers = extract_numbers_as_list(file_names)
            file_names = sorted(file_names, key=extract_number)
            for j in range(len(file_numbers) - self.num_input):
                # Temperature input
                temperature_in = []
                time_input = []
                for k in range(self.num_input):
                    file_name = file_names[j + k]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    # Einmalig zu jedem Trainingsbeispiel Koordinaten mit anlegen als Dimension
                    if k == 0:
                        coord_in = data.iloc[:, :1].to_numpy()
                        coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        # Continuation Coordinates
                        if continuation is True:
                            # step_size = coord_in[0][1] - coord_in[0][0]
                            pot = np.ceil(np.log2(coord_in.shape[1]))
                            size = 2 ** (pot + 1)
                            pad_size = int(size - coord_in.shape[1])
                            coord_in = np.pad(coord_in, ((0, 0), (0, pad_size)), "constant")
                            # coord_in = np.array([x * step_size for x in range(int(size))])
                            # coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        temperature_in.append(coord_in)
                    temp_in = data.iloc[:, -1].to_numpy()
                    # Reshape soll Dimension des realen Teils wiederspiegeln -> im eindimenionsalen eine Dimension 1
                    temp_in = temp_in.reshape((1, temp_in.shape[0]))
                    current_time = np.array([file_numbers[j+k] / 100])  
                    # Continuation Temperature padding
                    if continuation is True:
                        pad_size = int(size - temp_in.shape[1])
                        temp_in = np.pad(temp_in, ((0, 0), (0, pad_size)), "constant")
                    temperature_in.append(temp_in)
                    time_in = np.ones_like(temp_in) * current_time
                    time_input.append(current_time.item())
                    if time_channel:
                       temperature_in.append(time_in)
                temperature_in = np.concatenate(temperature_in, axis=0)
                self.inputs_lst.append([dir_name, torch.from_numpy(temperature_in).to(dtype=torch.float32)])
                self.time_list.append(time_input)

                # Boundarys
                size = temperature_in.shape[1]
                last_index = len(self.inputs_lst) - 1
                temp_zone = temp_zones[:, :, i].flatten()
                condition = condition_input[:, i].flatten()
                thickness_input = np.expand_dims(thickness, 0)
                # timestep_input = np.expand_dims(timesteps[i], 0)
                material_input_exp = material_input[:, i].flatten()  # np.squeeze(material_input, axis=-1)
                # The number in the data shows the steps -> one step = 0.01s
                current_time = np.array([file_numbers[j] / 100])
                boundary = [temp_zone, condition, material_input_exp, thickness_input, current_time]

                if time_channel is True:
                    # time = torch.tensor(boundary[-1])
                    # time = time.expand(1, size)
                    # self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], time), dim=0)
                    boundary = boundary[0:-1]
                if start_temp_channel is True:
                    start_temperature = torch.tensor(condition[0])
                    start_temperature = start_temperature.expand(1, size)
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], start_temperature),
                                                               dim=0)
                    boundary[1] = condition[1:]
                if max_channel is True:
                    conditions = condition
                    for con in conditions:
                        c = torch.tensor(con)
                        c = c.expand(1, size)
                        self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], c), dim=0)
                    del boundary[1]
                # boundary temperature manuell für jedes Bild
                if boundary_temp_channel is True:
                    boundary_temps = len(boundary[0])
                    temps = int(boundary_temps / self.zone_numbers)
                    data_size = self.inputs_lst[0][1].shape[1]
                    start = 0
                    end = 0
                    if j <= 24:
                        start = 0
                        end = temps
                    elif 24 < j <= 51:
                        start = temps
                        end = temps * 2
                    elif 51 < j <= 66:
                        start = temps * 2
                        end = temps * 3
                    elif 66 < j <= 73:
                        start = temps * 3
                        end = temps * 4
                    elif 73 < j <= 74:
                        start = temps * 4
                        end = temps * 5
                    elif 74 < j <= 80:
                        start = temps * 5
                        end = temps * 6
                    elif 80 < j <= 84:
                        start = temps * 6
                        end = temps * 7
                    elif 84 < j <= 86:
                        start = temps * 7
                        end = temps * 8
                    left, right, air = boundary[0][start:end]
                    halfway = data_size // 2
                    first_half = np.linspace(left, air, halfway, endpoint=False)
                    second_half = np.linspace(air, right, data_size - halfway)
                    boundary_temp = torch.tensor(np.concatenate((first_half, second_half))).to(dtype=torch.float32)
                    boundary_temp = boundary_temp.reshape((1, boundary_temp.shape[0]))  
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], boundary_temp),
                                                               dim=0)
                    del boundary[0]
                # Temperature and Boundary combined
                boundary = np.concatenate(boundary).astype(float).reshape(1, -1)
                self.inputs_lst[last_index].append(torch.from_numpy(boundary).to(dtype=torch.float32))
                # self.inputs_lst.append((dir_name, torch.tensor(inp), torch.tensor(np.array(temperature_in))))

                # Temperature output
                # temperature_out = []
                for y in range(j + k + 1, j + k + self.num_output + 1):
                    file_name = file_names[y]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                    temp_out = data.iloc[:, -1].to_numpy()
                    # Berechne Anzahl Zeilen und Spalten am besten
                    temperature_out = temp_out.reshape((1, temp_out.shape[0]))
                    # temperature_out.append(temp_out)
                self.outputs_lst.append((dir_name, torch.tensor(np.array(temperature_out), dtype=torch.float32)))

        self.input_size = self.inputs_lst[0][1].shape
        self.input_count = len(self.inputs_lst)
        self.boundary_size = self.inputs_lst[0][2].shape
        self.output_size = self.outputs_lst[0][1].shape
        self.output_count = len(self.outputs_lst)

    def getDataFromFrame(self, df: pd.DataFrame):
        # Zones
        zones = []
        i = 2
        # count zones
        self.zone_numbers = 0
        while df.iloc[i, 0].startswith("zone"):
            zones.append(df.iloc[i + 1:i + 4, :].dropna(axis=1))
            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
            i += 4
            self.zone_numbers += 1
        self.zone_end = i - self.zone_numbers
        # Conditional Info
        # df.iloc[i:i+5]
        conditions = df.iloc[i:i+12, :]
        conditions.columns = self.col_names[:]
        i += 12
        # Material info (Annahme: 1 Material)
        material = df.iloc[i:i + 9, :].dropna(axis=0)
        material.columns = self.col_names[:]
        return zones, conditions, material

    def getInputParamCount(self):
        return self.input_size

    def collate_function(self, batch):
        in_d, out_d = zip(*batch)
        _, in_tensor, in_bound = zip(*in_d)
        _, out_tensor = zip(*out_d)
        in_temp_tensors = torch.stack(in_tensor)
        in_bound_tensors = torch.stack(in_bound)
        out_tensors = torch.stack(out_tensor)
        return (in_temp_tensors, in_bound_tensors), out_tensors

    def __len__(self):
        return self.input_count

    def __getitem__(self, index):
        # Read the temperature solutions
        params = self.inputs_lst[index]
        solution = self.outputs_lst[index]
        return params, solution


class SingleStepDataset1D(Dataset):
    """
    Returns a batch of x, y tuples of the shape
    x: [batch_size, channels, data_size] for t
    y: [batch_size, 1, data_size] for t + 1
    """

    def __init__(self, dataPath, thickness) -> None:
        super().__init__()
        # self.time_per_step = 400
        self.solutionPath = os.path.join(dataPath, "Trainingsdaten_1D_testing")
        self.data = os.listdir(self.solutionPath)
        # First peek into the data
        data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress_1D_testing.xlsx'))

        # Load the zone values and name/thickness vals
        self.df = data.parse(sheet_name=str(thickness) + 'mm', skiprows=0)
        cols1 = list(self.df.columns)
        names = list(self.df.iloc[0, 1:])
        self.col_names = ["Attribute"]
        self.col_names.extend(names)
        self.zone_end = None
        zones, conditions, material = self.getDataFromFrame(self.df)

        # create a numpy array with all the zones as input for the neural net
        tmp_zones = []
        for zone in zones:
            z = zone.iloc[:, 1:].to_numpy(dtype=float)
            tmp_zones.append(z)
        temp_zones = np.asarray(tmp_zones)

        # Create numpy arrays from the other conditions
        condition_input = conditions.iloc[:, 1:].to_numpy()
        material_input = material.iloc[:, 1:].to_numpy()
        material_input[0] = material_embedding(material_input[0])
        # thick = self.df.iloc[1, 1:].to_numpy()
        # print('thickness array:',thickness)
        # timesteps in s
        timesteps = self.df.iloc[35, 1:].to_numpy()

        # Add all the inputs together for the different simulation runs
        self.inputs_lst = []
        self.outputs_lst = []
        # dir_base_name = "TD_10_{}"
        dir_base_name = "td" + str(thickness) + "_{}"
        
        for i in range(temp_zones.shape[-1]):
            dir_name = dir_base_name.format(i + 1)
            file_names = []
            file_numbers = []
            # extract file information
            for file in os.listdir(os.path.join(self.solutionPath, dir_name)):
                file_names.append(file)
            file_numbers = extract_numbers_as_list(file_names)
            file_names = sorted(file_names, key=extract_number)
            for j in range(len(file_numbers)):
                # Boundarys
                temp_zone = temp_zones[:, :, i].flatten()
                condition = condition_input[:, i].flatten()
                thickness_input = np.expand_dims(thickness, 0)
                # timestep_input = np.expand_dims(timesteps[i], 0)
                material_input_exp = material_input[:, i].flatten()  # np.squeeze(material_input, axis=-1)
                # The number in the data shows the steps -> one step = 0.01s
                current_time = np.array([file_numbers[j] / 100])
                inp = np.concatenate(
                    (temp_zone, condition, material_input_exp, thickness_input, current_time)).astype(float).reshape(1, -1)
                self.inputs_lst.append([dir_name, torch.from_numpy(inp).to(dtype=torch.float32)])

                # Temperature output
                file_name = file_names[j]
                data_path = os.path.join(self.solutionPath, dir_name)
                data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "temp[K]"])
                temperature_in = data.iloc[:, -1].to_numpy()
                # Berechne Anzahl Zeilen und Spalten am besten
                temperature_out = temperature_in.reshape((1, temperature_in.shape[0])).astype(float)
                self.outputs_lst.append([dir_name, torch.from_numpy(temperature_out).to(dtype=torch.float32)])

        self.input_size = self.inputs_lst[0][1].shape
        self.input_count = len(self.inputs_lst)
        self.output_size = self.outputs_lst[0][1].shape
        self.output_count = len(self.outputs_lst)

    def getDataFromFrame(self, df: pd.DataFrame):
        # Zones
        zones = []
        i = 2
        # count zones
        j = 0
        while df.iloc[i, 0].startswith("zone"):
            zones.append(df.iloc[i + 1:i + 4, :].dropna(axis=1))
            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
            i += 4
            j += 1
        self.zone_end = i - j
        # Conditional Info
        # df.iloc[i:i+5]
        conditions = df.iloc[i:i+12, :]
        conditions.columns = self.col_names[:]
        i += 12
        # Material info (Annahme: 1 Material)
        material = df.iloc[i:i + 9, :].dropna(axis=0)
        material.columns = self.col_names[:]
        return zones, conditions, material

    def getInputParamCount(self):
        return self.input_size

    def collate_function(self, batch):
        in_d, out_d = zip(*batch)
        _, in_tensor = zip(*in_d)
        _, out_tensor = zip(*out_d)
        in_tensors = torch.stack(in_tensor)
        out_tensors = torch.stack(out_tensor)
        return in_tensors, out_tensors

    def __len__(self):
        return self.input_count

    def __getitem__(self, index):
        # Read the temperature solutions
        params = self.inputs_lst[index]
        solution = self.outputs_lst[index]
        return params, solution


def material_embedding(material: str):
    ret = []
    known_materials = [None, "material 1"]
    for mat in material:
        ret.append(known_materials.index(mat))

    #ret = np.zeros_like(inputs)
    #for i, inp in enumerate(inputs):
    #    assert inp in known_materials, f"Unknown material: {inp}"
    #    ret[i] = known_materials.index(inp)
    return np.array(ret)


class SingleStepDataset2D(Dataset):
    """
    Returns a batch of x, y tuples of the shape
    x: [batch_size, channels, data_size, data_size] for t
    y: [batch_size, 1, data_size, data_size] for t + 1
    """
    def __init__(self, dataPath, thickness, num_input=1, data_start=0, data_end=40, continuation=False, time_channel=False, start_temp_channel=False) -> None:
        super().__init__()
        print('Glasdicke ist gleich:',thickness)
        self.inputs = []
        self.outputs = []
        self.time_list = []
        self.num_input = num_input
        self.num_output = 1
        self.data_start = data_start
        self.data_end = data_end
        self.solutionPath = os.path.join(dataPath, "Temperature")
        self.data = os.listdir(self.solutionPath )
        # First peek into the data
        data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress.xlsx'))

        # Load the zone values and name/thickness vals
        self.df= data.parse(sheet_name=str(thickness)+'mm', skiprows=0)
        cols1 = list(self.df.columns)
        names = list(self.df.iloc[0,1:])
        #print('names', names)
        self.col_names = ["Attribute"]
        self.col_names.extend(names)
        zones, conditions = self.getDataFromFrame(self.df)
        #print('Conditions:',conditions)
        #print('Zonen:', zones)
        #create a numpy array with all the zones as input for the neural net
        tmp_zones = []
        #print('Alle zonen', zones)
        for zone in zones:
            #print('zone_initial', zone)
            zone = zone.iloc[:,1:].to_numpy(dtype=int)
            #print('zone_clean', zone)
            tmp_zones.append(zone)
        temp_zones = np.asarray(tmp_zones)
        #print('Zonen:', temp_zones)
        #Create numpy arrays from the dataframes
        condition_input =  conditions.iloc[1:, ].to_numpy()
        #material_input = material.iloc[:,1:].to_numpy()
        #material_input[0] = material_embedding(material_input[0])
        #thickness= self.df.iloc[1,1:].to_numpy()
        #distance = self.df.iloc[107,1:].to_numpy()
        distance = []
        for o in range(71):
            distance.append(o)
        for o in range(80,151,5):
            distance.append(o)
    

        #Add all the inputs together for the different simulation runs
        # self.inputs_lst = []
        # #dir_base_name = "TD_10_{}"
        # dir_base_name = "td"+str(thickness)+"_{}"
        # for i in range(temp_zones.shape[-1]):
        #     dir_name = dir_base_name.format(i+1)
        #     for dist in distance:
        #         temp_zone = temp_zones[:,:,i].flatten()
        #         #print('Zone T:', temp_zone)
        #         condition = np.expand_dims(condition_input[i], 0)
        #         #print('Start T:',condition)
        #         thickness_input = np.expand_dims(thickness, 0)
        #         #print('thick:', thickness_input)
        #         z_pos_input = np.expand_dims(dist, 0)
        #         #print('timestep:', timestep_input)
        #         #material_input_exp = np.squeeze(material_input, axis=-1)
        #         input = np.concatenate((temp_zone, condition, thickness_input, z_pos_input)).astype(float)
        #         self.inputs_lst.append( (dir_name, torch.tensor(input)) )
        # #print('Input_lst:', self.inputs_lst)
        # self.input_count = self.inputs_lst[0][1].shape[0]
        # self.sim_count = len(self.inputs_lst)
        # print(f"Number of experiments: {self.sim_count}")
        #print(f"Input param size: {self.input_count}")
        
        
        # Add all the inputs together for the different simulation runs
        self.inputs_lst = []
        self.outputs_lst = []
        # dir_base_name = "TD_10_{}"
        
        dir_base_name = "td3.8_{}"
        for i in range(data_start, data_end):  #temp_zones.shape[-1]
            dir_name = dir_base_name.format(i+1)
            file_names = []
            file_numbers = []
            # extract file information 
            for file in os.listdir(os.path.join(self.solutionPath, dir_name)):
                file_names.append(file)
            file_numbers = extract_numbers_as_list(file_names)
            file_names = sorted(file_names, key=extract_number)
            for j in range(len(file_numbers) - self.num_input):
                # Temperature input
                temperature_in = []
                time_input = []
                for k in range(self.num_input):
                    file_name = file_names[j + k]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "y[mm]", "temp[K]"])
                    # Einmalig zu jedem Trainingsbeispiel Koordinaten mit anlegen als Dimension
                    if k == 0:
                        x_coord_in = data.iloc[:, 0].to_numpy()
                        x_coord_in = x_coord_in.reshape(1, 20, -1)
                        y_coord_in = data.iloc[:, 1].to_numpy()
                        y_coord_in = y_coord_in.reshape(1, 20, -1)
                        # Continuation Coordinates
                        if continuation is True:
                            # step_size = coord_in[0][1] - coord_in[0][0]
                            pot = np.ceil(np.log2(x_coord_in.shape[2]))
                            size = 2 ** (pot + 1)
                            
                            pad_size = int(size - x_coord_in.shape[2])
                            x_coord_in = np.pad(x_coord_in, ((0, 0), (0, 0), (0, pad_size)), "constant")
                            y_coord_in = np.pad(y_coord_in, ((0, 0), (0, 0), (0, pad_size)), "constant")
                            # coord_in = np.array([x * step_size for x in range(int(size))])
                            # coord_in = coord_in.reshape((1, coord_in.shape[0]))
                        temperature_in.append(x_coord_in)
                        temperature_in.append(y_coord_in)
                    temp_in = data.iloc[:, -1].to_numpy()
                    # Reshape soll Dimension des realen Teils wiederspiegeln -> im eindimenionsalen eine Dimension 1
                    temp_in = temp_in.reshape(1, 20, -1)
                    current_time = np.array([file_numbers[j+k] / 100])  
                    # Continuation Temperature padding
                    if continuation is True:
                        pad_size = int(size - temp_in.shape[2])
                        temp_in = np.pad(temp_in, ((0, 0), (0, 0), (0, pad_size)), "constant")
                    temperature_in.append(temp_in)
                    time_in = np.ones_like(temp_in) * current_time
                    time_input.append(current_time.item())
                    if time_channel:
                       temperature_in.append(time_in)
                temperature_in = np.concatenate(temperature_in, axis=0)
                self.inputs_lst.append([dir_name, torch.from_numpy(temperature_in).to(dtype=torch.float32)])
                self.time_list.append(time_input)

                # Boundarys
                size = temperature_in.shape[1]
                last_index = len(self.inputs_lst) - 1
                temp_zone = temp_zones[:, :, i].flatten()
                condition = np.array([condition_input[i]])
                thickness_input = np.expand_dims(thickness, 0)
                # timestep_input = np.expand_dims(timesteps[i], 0)
                # material_input_exp = material_input[:, i].flatten()  # np.squeeze(material_input, axis=-1)
                # The number in the data shows the steps -> one step = 0.01s
                current_time = np.array([file_numbers[j] / 100])
                # boundary = [temp_zone, condition, material_input_exp, thickness_input, current_time]
                boundary = [temp_zone, condition, thickness_input, current_time]

                if time_channel is True:
                    # time = torch.tensor(boundary[-1])
                    # time = time.expand(1, size)
                    # self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], time), dim=0)
                    boundary = boundary[0:-1]
                if start_temp_channel is True:
                    start_temperature = torch.tensor(condition[0])
                    start_temperature = start_temperature.expand(1, size)
                    self.inputs_lst[last_index][1] = torch.cat((self.inputs_lst[last_index][1], start_temperature),
                                                               dim=0)
                    boundary[1] = condition[1:]
                # Temperature and Boundary combined
                boundary = np.concatenate(boundary).astype(float).reshape(1, -1)
                self.inputs_lst[last_index].append(torch.from_numpy(boundary).to(dtype=torch.float32))
                # self.inputs_lst.append((dir_name, torch.tensor(inp), torch.tensor(np.array(temperature_in))))

                # Temperature output
                # temperature_out = []
                for y in range(j + k + 1, j + k + self.num_output + 1):
                    file_name = file_names[y]
                    data_path = os.path.join(self.solutionPath, dir_name)
                    data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "y[mm]", "temp[K]"])
                    temp_out = data.iloc[:, -1].to_numpy()
                    # Berechne Anzahl Zeilen und Spalten am besten
                    temperature_out = temp_out.reshape(1, 20, -1)
                    # temperature_out.append(temp_out)
                self.outputs_lst.append((dir_name, torch.tensor(np.array(temperature_out), dtype=torch.float32)))
                
        self.sim_count = len(self.inputs_lst)
        self.input_count = self.inputs_lst[0][1].shape
        
        
        # for i in range(len(self.inputs_lst) - 1):
        #     print("data")
        #     input = self.inputs_lst[i]
        #     #print(input[0])
        #     params = input[1]
        #     z_pos = params[-1]
        #     file_name = f"glass_temp_{int(z_pos)*400}.0"
        #     data_path= os.path.join(self.solutionPath, input[0])
        #     data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "y[mm]", "temp[K]"])
        #     data = data.iloc[:,-1].to_numpy()
        #     data = data.reshape((1, 20, 16901))
        #     solution = torch.tensor(data)
        #     # params = params.type(torch.float)
        #     # params = torch.nan_to_num(params, nan=-1)
        #     in_data = solution.type(torch.float)
        #     
        #     input = self.inputs_lst[i+1]
        #     params = input[1]
        #     z_pos = params[-1]
        #     file_name = f"glass_temp_{int(z_pos*400)}.0"
        #     data_path= os.path.join(self.solutionPath, input[0])
        #     data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "y[mm]", "temp[K]"])
        #     data = data.iloc[:,-1].to_numpy()
        #     data = data.reshape((1, 20, 16901))
        #     solution = torch.tensor(data)
        #     # params = params.type(torch.float)
        #     # params = torch.nan_to_num(params, nan=-1)
        #     out_data = solution.type(torch.float)
        #     
        #     self.inputs.append(in_data)
        #     self.outputs.append(out_data)
        # print("Dataset is ready!")
            
        
    def getDataFromFrame(self, df:pd.DataFrame):   
        #Zones
        zones = []
        #print('data_frame:',df)
        i = 2
        while df.iloc[i,0].startswith("zone"):
            zones.append(df.iloc[i+1:i+13,:].dropna(axis=1))
            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
            i+=13
        # Conditional Info
        #df.iloc[i:i+5]
        #df.iloc[i]
        #print('Conditions:',df.iloc[i,:][1])
        #print(self.col_names[:])
        conditions = df.iloc[i,:]
        conditions.columns = self.col_names[:]
        #i+=6
        #Material info
        #material = df.iloc[i:i+9,0:2] #.dropna(axis=0)
        #material.columns = ["Attribute", "Value"]
        return zones, conditions 

    def getInputParamCount(self):
        return self.input_count

    def __len__(self):
        return self.sim_count 
    
    def __getitem__(self, index):
        return self.inputs_lst[index], self.outputs_lst[index]

    def collate_function(self, batch):
        in_d, out_d = zip(*batch)
        _, in_tensor, in_bound = zip(*in_d)
        _, out_tensor = zip(*out_d)
        in_temp_tensors = torch.stack(in_tensor)
        in_bound_tensors = torch.stack(in_bound)
        out_tensors = torch.stack(out_tensor)
        return (in_temp_tensors, in_bound_tensors), out_tensors


#class GlassDataset(Dataset):
#    def __init__(self, dataPath, thickness) -> None:
#        super().__init__()
#        self.solutionPath = os.path.join(dataPath, "Temperature")
#        self.data = os.listdir(self.solutionPath )
#        # First peek into the data
#        data = pd.ExcelFile(os.path.join(dataPath, 'DoE_Therm_stress.xlsx'))
#        #peek first ten rows
#        # Load the zone values and name/thickness vals
#        self.df= data.parse(sheetname='10mm', skiprows=0)
#        cols1 = list(self.df.columns)
#        names = list(self.df.iloc[0,1:])
#        self.col_names = ["Attribute"]
#        self.col_names.extend(names[0:-1])
#        zones, conditions, material = self.getDataFromFrame(self.df)
#        #create a numpy array with all the zones as input for the neural net
#        tmp_zones = []
#        for zone in zones:
#            zone = zones[-1].iloc[:,1:].to_numpy(dtype=int)
#            tmp_zones.append(zone)
#        temp_zones = np.asarray(tmp_zones)
#
#        #Create numpy arrays from the dataframes
#        condition_input =  conditions.iloc[:,1:].to_numpy()
#        material_input = material.iloc[:,1:].to_numpy()
#        material_input[0] = material_embedding(material_input[0])
#        thickness= self.df.iloc[1,1:].to_numpy()
#        #TODO add the thickness as a parameter to the input
#
#        #Add all the inputs together for the different simuilation runs
#        self.inputs_lst = []
#        for i in range(temp_zones.shape[-1]):
#            temp_zone = temp_zones[:,:,i].flatten()
#            condition = condition_input[:,i].flatten()
#            thickness_input = np.expand_dims(thickness, 0)
#            material_input_exp = np.squeeze(material_input, axis=-1)
#            input = np.concatenate((temp_zone, condition, material_input_exp, thickness_input )).astype(float)
#            self.inputs_lst.append(torch.tensor(input))
#        self.input_count= self.inputs_lst[0].shape[0]
#        self.sim_count = len(self.inputs_lst)
#        print(f"Number of experiments: {self.sim_count}")
#        print(f"Input param size: {self.input_count}")
#
#    def getDataFromFrame(self, df:pd.DataFrame):
#        #Zones
#        zones = []
#        i = 2
#        while df.iloc[i,0].startswith("zone"):
#            zones.append(df.iloc[i+1:i+12,:].dropna(axis=1))
#            zones[-1].columns = self.col_names[:len(zones[-1].columns)]
#            i+=12
#        # Conditional Info
#        df.iloc[i:i+5]
#        conditions = df.iloc[i:i+5,:-1]
#        conditions.columns = self.col_names[:]
#        i+=6
#        #Material info
#        material = df.iloc[i:i+9,0:2] #.dropna(axis=0)
#        material.columns = ["Attribute", "Value"]
#        return zones, conditions, material
#
#    def getInputParamCount(self):
#        return self.input_count
#
#
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, index):
#        # Read the temperature solutions
#        input = self.inputs_lst[index]
#        data_path = os.path.join(self.solutionPath, self.data[index])
#        timesteps = os.listdir(data_path)
#        pos_data = []
#        for file_name in os.listdir(data_path):
#            data = pd.read_csv(os.path.join(data_path, file_name), header=None, names=["x[mm]", "y[mm]", "temp[K]"])
#            data = data.iloc[:,-1].to_numpy()
#            data = data.reshape((21, 2001))
#            pos_data.append(np.expand_dims(data, 0))
#        solution = torch.tensor(np.concatenate(pos_data, axis=0))
#        unorm = transforms.Normalize(mean=[-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616],
#                             std=[1/0.2470, 1/0.2435, 1/0.2616])
#        input = input.type(torch.float)
#        input = torch.nan_to_num(input, nan=-1)
#        solution = solution.type(torch.float)
#        return input, solution
    

if __name__ == '__main__':
    os.chdir('..')
    working_dir = os.getcwd()
    data_path = os.path.join(working_dir, "Models/Trainingsdaten/")
    print(os.listdir(data_path))
    d_path = os.path.join(data_path, 'Trainingsdaten_1D_light_complex')
    o_path = os.path.join(data_path, 'DoE_Therm_stress_1D_light_complex.xlsx')
    ds = SingleStepMultiTempDataset1D(dataPath=d_path, overviewPath=o_path, thickness=3.8, size=50, num_input=1,
                                      max_channel=False, time_channel=True, start_temp_channel=False, boundary_temp_channel=False, continuation=False, start=1)
    # MultiStepMultiTempDataset1D(dataPath=d_path, overviewPath=o_path, thickness=3.8, size=50, num_input=4, difference=0,
    #                                   max_channel=False, time_channel=True, start_temp_channel=False, boundary_temp_channel=False, continuation=False)
    # input_vector_3 = torch.randn(batch_size, multi_ds.input_size[0], multi_ds.input_size[1])
    i, s = ds.__getitem__(index=2)
    #train_dataloader = DataLoader(
    #                ds,
    #                batch_size=24,
    #                num_workers=0,
    #                shuffle=False)
    #data_iter = iter(train_dataloader)
    #print(next(data_iter))
