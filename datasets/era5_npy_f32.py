from torch.utils.data import Dataset
import numpy as np
import io
import json
import pandas as pd
import os
from multiprocessing import shared_memory
import multiprocessing
import copy
import queue
import torch


Years = {
    'train': ['1979-01-01 00:00:00', '2015-12-31 23:00:00'],
    'valid': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'test': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
}

multi_level_vnames = [
    "z", "t", "q", "r", "u", "v", "vo", "pv",
]
single_level_vnames = [
    "t2m", "u10", "v10", "tcc", "tp", "tisr",
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}

height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
# height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


class era5_npy_f32(Dataset):
    def __init__(self, data_dir='./data', split='train', **kwargs) -> None:
        super().__init__()

        self.length = kwargs.get('length', 1)
        self.file_stride = kwargs.get('file_stride', 6)
        self.sample_stride = kwargs.get('sample_stride', 1)
        self.output_meanstd = kwargs.get("output_meanstd", False)
        self.use_diff_pos = kwargs.get("use_diff_pos", False)
        self.rm_equator = kwargs.get("rm_equator", False)
        Years_dict = kwargs.get('years', Years)

        self.pred_length = kwargs.get("pred_length", 0)
        self.inference_stride = kwargs.get("inference_stride", 6)
        self.train_stride = kwargs.get("train_stride", 6)
        self.use_gt = kwargs.get("use_gt", True)
        self.data_save_dir = kwargs.get("data_save_dir", None)

        self.save_single_level_names = kwargs.get("save_single_level_names", [])
        self.save_multi_level_names = kwargs.get("save_multi_level_names", [])

        vnames_type = kwargs.get("vnames", {})
        self.single_level_vnames = vnames_type.get('single_level_vnames', [])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z','q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]

        self.select_row = [i for i in range(721)]
        if self.rm_equator:
            del self.select_row[360]
        self.split = split
        self.data_dir = data_dir
        years = Years_dict[split]
        self.init_file_list(years)

        self._get_meanstd()
        self.mean, self.std = self.get_meanstd()
        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)
        dim = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)

        self.index_dict1 = {}
        self.index_dict2 = {}
        i = 0
        for vname in self.single_level_vnames:
            self.index_dict1[(vname, 0)] = i
            i += 1
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_dict1[(vname, height)] = i
                i += 1

        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()

        self.index_queue.cancel_join_thread() 
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}

        self.compound_data_queue_num = 8

        self.lock = multiprocessing.Lock()
        if self.rm_equator:
            self.a = np.zeros((dim, 720, 1440), dtype=np.float32)
        else:
            self.a = np.zeros((dim, 721, 1440), dtype=np.float32)

        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.a.nbytes)
            shm.unlink()
            self.sharedmemory_list.append(shm)

        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))

        self._workers = []

        for _ in range(40):
            w = multiprocessing.Process(
                target=self.load_data_process)
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._workers.append(w)
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)

    def init_file_list(self, years):
        time_sequence = pd.date_range(years[0],years[1],freq=str(self.file_stride)+'H') #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.file_list= [os.path.join(str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        self.single_file_list= [os.path.join('single/'+str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]

    def _get_meanstd(self):
        with open('./datasets/mean_std.json',mode='r') as f:
            multi_level_mean_std = json.load(f)
        with open('./datasets/mean_std_single.json',mode='r') as f:
            single_level_mean_std = json.load(f)
        self.mean_std = {}
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        self.mean_std['mean'] = multi_level_mean_std['mean']
        self.mean_std['std'] = multi_level_mean_std['std']
        for vname in self.single_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
        for vname in self.multi_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]

    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()

            if (job_pid, idx) in recorder_dict:
                # recorder_dict[(job_pid, idx)][(vname, height)] = 1
                recorder_dict[(job_pid, idx)] += 1
            else:
                recorder_dict[(job_pid, idx)] = 1
            if recorder_dict[(job_pid, idx)] == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))

    def get_data(self, idxes):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        return_data = []
        for idx in idxes:
            for vname in self.single_level_vnames:
                self.index_queue.put((job_pid, idx, vname, 0))
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    self.index_queue.put((job_pid, idx, vname, height))
            idx = self.compound_data_queue_dict[job_pid].get()
            b -= self.mean.numpy()[:, np.newaxis, np.newaxis]
            b /= self.std.numpy()[:, np.newaxis, np.newaxis]
            return_data.append(copy.deepcopy(b))
            
        return return_data

    def load_data_process(self):
        while True:
            job_pid, idx, vname, height = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            if vname in self.single_level_vnames:
                file = self.single_file_list[idx]
                url = f"{self.data_dir}/{file}-{vname}.npy"
            elif vname in self.multi_level_vnames:
                file = self.file_list[idx]
                url = f"{self.data_dir}/{file}-{vname}-{height}.0.npy"
            b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            unit_data = np.load(url)
            # unit_data = unit_data[np.newaxis, :, :]
            if self.rm_equator:
                b[self.index_dict1[(vname, height)], :360] = unit_data[:360]
                b[self.index_dict1[(vname, height)], 360:] = unit_data[361:]
            else:
                b[self.index_dict1[(vname, height)], :] = unit_data[:]
            del unit_data
            self.unit_data_queue.put((job_pid, idx, vname, height))

    def __len__(self):

        if self.split != "test":
            data_len = (len(self.file_list) - (self.length - 1) * self.sample_stride) // (self.train_stride // self.sample_stride // self.file_stride)
        elif self.use_gt:
            data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
            data_len -= self.pred_length * self.sample_stride + 1
            data_len = (data_len + max(self.inference_stride // self.sample_stride // self.file_stride, 1) - 1) // max(self.inference_stride // self.sample_stride // self.file_stride, 1)
        else:
            data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
            data_len = (data_len + max(self.inference_stride // self.sample_stride // self.file_stride, 1) - 1) // max(self.inference_stride // self.sample_stride // self.file_stride, 1)

        return data_len
    
    def get_meanstd(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname][self.height_level_indexes])
            return_data_std.append(self.mean_std['std'][vname][self.height_level_indexes])

        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0]), torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])

    def __getitem__(self, index):
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        if self.split == "test":
            index = index * max(self.inference_stride // self.sample_stride // self.file_stride, 1)
        else:
            index = index * (self.train_stride // self.sample_stride // self.file_stride)
        array_seq = self.get_data([index, index + self.sample_stride, index + (self.length-1) * self.sample_stride])
        tar_idx = np.array([index + self.sample_stride * (self.length - 1)])
        return array_seq, tar_idx
