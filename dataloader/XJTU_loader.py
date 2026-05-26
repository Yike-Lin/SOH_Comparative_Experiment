from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from utils.Scaler import Scaler

class XJTUDdataset():
    def __init__(self,args):
        super(XJTUDdataset).__init__()
        self.root = 'data/XJTU'
        self.max_capacity = 2.0
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch = args.batch
        self.batch_size = args.batch_size
        self.input_type = args.input_type

    def _parse_profile(self, cycle_i_data):
        time = cycle_i_data['relative_time_min']
        current = cycle_i_data['current_A']
        voltage = cycle_i_data['voltage_V']
        temperature = cycle_i_data['temperature_C']
        cycle_i = np.concatenate([time, current, voltage, temperature], axis=0)
        return cycle_i

    def _parse_cycle(self, cycle_i_data):
        cycle_i = self._parse_profile(cycle_i_data)
        capacity = float(cycle_i_data['capacity'][0][0])
        return cycle_i, capacity



    def _parser_mat_data(self,battery_i_mat):
        '''
        :param battery_i_mat: shape:(1,len)
        :return: np.array
        '''
        data = []
        label = []
        for i in range(battery_i_mat.shape[1]):
            cycle_i_data = battery_i_mat[0,i]
            cycle_i, capacity = self._parse_cycle(cycle_i_data)
            label.append(capacity)
            data.append(cycle_i)
        data = np.array(data,dtype=np.float32)
        label = np.array(label,dtype=np.float32)
        print(data.shape,label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data,soh

    def _parser_full_mat_data(self, battery_i_mat, battery_id=None, return_meta=False):
        '''
        :param battery_i_mat: shape:(1,len) with field `cycles`
        :return: np.array
        '''
        cycles = battery_i_mat['cycles']
        data = []
        label = []
        meta = []
        for i in range(cycles.shape[1]):
            cycle_i_data = cycles[0, i]
            charge_data = cycle_i_data['charge_data'][0, 0]
            discharge_data = cycle_i_data['discharge_data'][0, 0]
            charge_i = self._parse_profile(charge_data)
            discharge_i = self._parse_profile(discharge_data)
            capacity = float(cycle_i_data['capacity'][0][0])
            label.append(capacity)
            data.append(np.concatenate([charge_i, discharge_i], axis=0))
            if return_meta:
                meta.append({
                    'battery_id': battery_id,
                    'cycle_id': i + 1,
                    'soh': capacity / self.max_capacity,
                })
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        print(data.shape, label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        if return_meta:
            return data, soh, pd.DataFrame(meta)
        return data, soh

    def _parser_full_discharge_mat_data(self, battery_i_mat):
        '''
        Parse only the discharge profile from full charge/discharge files.
        Output shape: (N, 4, 128)
        '''
        cycles = battery_i_mat['cycles']
        data = []
        label = []
        for i in range(cycles.shape[1]):
            cycle_i_data = cycles[0, i]
            discharge_data = cycle_i_data['discharge_data'][0, 0]
            discharge_i = self._parse_profile(discharge_data)
            capacity = float(cycle_i_data['capacity'][0][0])
            label.append(capacity)
            data.append(discharge_i)
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        print(data.shape, label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def _parser_full_charge_mat_data(self, battery_i_mat):
        '''
        Parse only the charge profile from full charge/discharge files.
        Output shape: (N, 4, 128)
        '''
        cycles = battery_i_mat['cycles']
        data = []
        label = []
        for i in range(cycles.shape[1]):
            cycle_i_data = cycles[0, i]
            charge_data = cycle_i_data['charge_data'][0, 0]
            charge_i = self._parse_profile(charge_data)
            capacity = float(cycle_i_data['capacity'][0][0])
            label.append(capacity)
            data.append(charge_i)
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        print(data.shape, label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def _parser_dual_mat_data(self, charge_battery_mat, partial_battery_mat):
        '''
        Align charge and partial_charge cycle by cycle, then concatenate them into 8 channels.
        '''
        charge_cycles = charge_battery_mat.shape[1]
        partial_cycles = partial_battery_mat.shape[1]
        n_cycles = min(charge_cycles, partial_cycles)
        if charge_cycles != partial_cycles:
            print(f'warning: charge cycles ({charge_cycles}) != partial cycles ({partial_cycles}), truncate to {n_cycles}')

        data = []
        label = []
        for i in range(n_cycles):
            charge_cycle = charge_battery_mat[0, i]
            partial_cycle = partial_battery_mat[0, i]
            charge_i, charge_capacity = self._parse_cycle(charge_cycle)
            partial_i, partial_capacity = self._parse_cycle(partial_cycle)
            if not np.isclose(charge_capacity, partial_capacity, atol=1e-5):
                raise ValueError(
                    f'charge and partial_charge labels mismatch at cycle {i + 1}: '
                    f'{charge_capacity} vs {partial_capacity}'
                )
            data.append(np.concatenate([charge_i, partial_i], axis=0))
            label.append(charge_capacity)

        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        print(data.shape, label.shape)

        scaler = Scaler(data)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def _encapsulation(self,train_x,train_y,test_x,test_y):
        '''
        Encapsulate the numpy.array into DataLoader
        :param train_x: numpy.array
        :param train_y: numpy.array
        :param test_x: numpy.array
        :param test_y: numpy.array
        :return:
        '''
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=self.seed)
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=self.batch_size, shuffle=True,
                                  drop_last=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    def _get_raw_data(self,path,test_battery_id,parser_fn=None):
        if parser_fn is None:
            parser_fn = self._parser_mat_data
        mat = loadmat(path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1][0]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = parser_fn(test_battery)
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1][0]
            x, y = parser_fn(train_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        return self._encapsulation(train_x, train_y, test_x, test_y)

    def _get_full_raw_data(self, path, test_battery_id):
        mat = loadmat(path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = self._parser_full_mat_data(test_battery)
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1]
            x, y = self._parser_full_mat_data(train_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        return self._encapsulation(train_x, train_y, test_x, test_y)

    def get_charge_data(self,test_battery_id=1):
        print('----------- load charge data from full charge/discharge files -------------')
        file_name = f'Batch{self.batch}_full.mat'
        self.full_path = os.path.join(self.root, 'full', file_name)
        if not os.path.exists(self.full_path):
            available = sorted(
                f for f in os.listdir(os.path.join(self.root, 'full'))
                if f.lower().endswith('_full.mat')
            )
            raise FileNotFoundError(
                f'full data file not found for batch {self.batch}: {self.full_path}. '
                f'Available full files: {available}'
            )
        mat = loadmat(self.full_path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = self._parser_full_charge_mat_data(test_battery)
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1]
            x, y = self._parser_full_charge_mat_data(train_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train':train_loader,
                     'test':test_loader,
                     'valid':valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict


    def get_partial_data(self,test_battery_id=1):
        print('----------- load partial_charge data -------------')
        file_name = f'batch-{self.batch}_3.7-4.1.mat'
        if self.batch == 6:
            file_name = f'batch-{self.batch}_3.9-4.19.mat'
        self.partial_path = os.path.join(self.root, 'partial_charge', file_name)
        train_loader, valid_loader, test_loader = self._get_raw_data(
            path=self.partial_path,
            test_battery_id=test_battery_id,
            parser_fn=self._parser_mat_data
        )
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('----------------  finished !  --------------------')
        return data_dict

    def get_discharge_data(self, test_battery_id=1):
        print('----------- load discharge data from full charge/discharge files -------------')
        file_name = f'Batch{self.batch}_full.mat'
        self.full_path = os.path.join(self.root, 'full', file_name)
        if not os.path.exists(self.full_path):
            available = sorted(
                f for f in os.listdir(os.path.join(self.root, 'full'))
                if f.lower().endswith('_full.mat')
            )
            raise FileNotFoundError(
                f'full data file not found for batch {self.batch}: {self.full_path}. '
                f'Available full files: {available}'
            )
        mat = loadmat(self.full_path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = self._parser_full_discharge_mat_data(test_battery)
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1]
            x, y = self._parser_full_discharge_mat_data(train_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict

    def get_full_data(self, test_battery_id=1):
        print('----------- load full charge/discharge data -------------')
        file_name = f'Batch{self.batch}_full.mat'
        self.full_path = os.path.join(self.root, 'full', file_name)
        if not os.path.exists(self.full_path):
            available = sorted(
                f for f in os.listdir(os.path.join(self.root, 'full'))
                if f.lower().endswith('_full.mat')
            )
            raise FileNotFoundError(
                f'full data file not found for batch {self.batch}: {self.full_path}. '
                f'Available full files: {available}'
            )
        train_loader, valid_loader, test_loader = self._get_full_raw_data(
            path=self.full_path,
            test_battery_id=test_battery_id,
        )
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict

    def get_full_arrays(self, test_battery_id=1):
        '''
        Return raw full-data arrays plus metadata for downstream analysis tools
        such as SHAP. The train split follows the same random_state as
        `_encapsulation`, so the sample set is consistent with training.
        '''
        print('----------- load full charge/discharge arrays -------------')
        file_name = f'Batch{self.batch}_full.mat'
        self.full_path = os.path.join(self.root, 'full', file_name)
        if not os.path.exists(self.full_path):
            available = sorted(
                f for f in os.listdir(os.path.join(self.root, 'full'))
                if f.lower().endswith('_full.mat')
            )
            raise FileNotFoundError(
                f'full data file not found for batch {self.batch}: {self.full_path}. '
                f'Available full files: {available}'
            )

        mat = loadmat(self.full_path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y, test_meta = self._parser_full_mat_data(
            test_battery,
            battery_id=test_battery_id,
            return_meta=True
        )

        train_x_list, train_y_list, train_meta_list = [], [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1]
            x, y, meta = self._parser_full_mat_data(
                train_battery,
                battery_id=id,
                return_meta=True
            )
            train_x_list.append(x)
            train_y_list.append(y)
            train_meta_list.append(meta)

        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        train_meta = pd.concat(train_meta_list, ignore_index=True)
        print('train data shape: ', train_x.shape, train_y.shape)

        all_train_index = np.arange(train_x.shape[0])
        train_index, valid_index = train_test_split(
            all_train_index,
            test_size=0.2,
            random_state=self.seed,
        )

        arrays = {
            'train': {
                'x': train_x[train_index],
                'y': train_y[train_index],
                'meta': train_meta.iloc[train_index].reset_index(drop=True),
                'index': np.asarray(train_index, dtype=int),
            },
            'valid': {
                'x': train_x[valid_index],
                'y': train_y[valid_index],
                'meta': train_meta.iloc[valid_index].reset_index(drop=True),
                'index': np.asarray(valid_index, dtype=int),
            },
            'test': {
                'x': test_x,
                'y': test_y,
                'meta': test_meta.reset_index(drop=True),
                'index': np.arange(test_x.shape[0], dtype=int),
            },
            'all_train': {
                'x': train_x,
                'y': train_y,
                'meta': train_meta.reset_index(drop=True),
                'index': np.arange(train_x.shape[0], dtype=int),
            },
        }
        print('-------------  finished !  ---------------')
        return arrays

    def get_charge_partial_data(self, test_battery_id=1):
        print('----------- load charge + partial_charge data -------------')
        charge_file_name = f'batch-{self.batch}.mat'
        partial_file_name = f'batch-{self.batch}_3.7-4.1.mat'
        if self.batch == 6:
            partial_file_name = f'batch-{self.batch}_3.9-4.19.mat'

        charge_path = os.path.join(self.root, 'charge', charge_file_name)
        partial_path = os.path.join(self.root, 'partial_charge', partial_file_name)

        charge_mat = loadmat(charge_path)['battery']
        partial_mat = loadmat(partial_path)['battery']

        battery_ids = list(range(1, charge_mat.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        test_x, test_y = self._parser_dual_mat_data(charge_mat[0, test_battery_id - 1][0],
                                                    partial_mat[0, test_battery_id - 1][0])
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            charge_battery = charge_mat[0, id - 1][0]
            partial_battery = partial_mat[0, id - 1][0]
            x, y = self._parser_dual_mat_data(charge_battery, partial_battery)
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict

    def _parser_xlsx(self,df_i):
        '''
        features dataframe
        :param df_i: shape:(N,C+1)
        :return:
        '''
        N = df_i.shape[0]
        x = np.array(df_i.iloc[:, :-1],dtype=np.float32)
        label = np.array(df_i['label'],dtype=np.float32).reshape(-1, 1)

        scaler = Scaler(x)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def get_features(self,test_battery_id=1):
        print('----------- load features -------------')
        file_name = f'batch-{self.batch}_features.xlsx'
        self.features_path = os.path.join(self.root, 'handcraft_features', file_name)
        df = pd.read_excel(self.features_path,sheet_name=None)
        sheet_names = list(df.keys())
        battery_ids = list(range(1, len(sheet_names)+1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')
        test_battery_df = pd.read_excel(self.features_path,sheet_name=test_battery_id-1,header=0)
        test_x,test_y = self._parser_xlsx(test_battery_df)
        print(f'test battery id: {test_battery_id}, test data shape: {test_x.shape}, {test_y.shape}')

        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            sheet_name = sheet_names[id-1]
            df_i = df[sheet_name]
            x, y = self._parser_xlsx(df_i)
            print(f'train battery id: {id}, {x.shape}, {y.shape}')
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x,axis=0)
        train_y = np.concatenate(train_y,axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('---------------  finished !  ----------------')
        return data_dict





if __name__ == '__main__':
    import argparse
    def get_args():

        parser = argparse.ArgumentParser(description='dataloader test')
        parser.add_argument('--random_seed',type=int,default=2023)
        # data
        parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU', 'MIT', 'CALCE'])
        parser.add_argument('--input_type', type=str, default='charge',
                            choices=['charge', 'partial_charge', 'handcraft_features'])
        parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
        parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (1, 1)])
        parser.add_argument('--batch_size', type=int, default=128)
        # the parameters for XJTU data
        parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5])

        args = parser.parse_args()
        return args

    args = get_args()
    data = XJTUDdataset(args)
    charge_data = data.get_charge_data()
    partial_data = data.get_partial_data()
    features = data.get_features()
