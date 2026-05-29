from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
import os
import re
from torchvision import transforms as A
from torch import nn
from torch.nn import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.tranforms = transforms

    def __call__(self, x, is_evs=False, is_flow=False):
        for t in self.tranforms:
            x = t(x, is_evs, is_flow)
        return x


class RandomCrop(object):
    def __init__(self, sensor_size, crop_size):
        self.h, self.w = int(sensor_size[0]), int(sensor_size[1])
        self.th, self.tw = int(crop_size[0]), int(crop_size[1])
        self.i = random.randint(0, self.h - self.th)
        self.j = random.randint(0, self.w - self.tw)

    def __call__(self, x, is_evs=False, is_flow=False):
        if is_evs:
            y_idx = (x[1] > self.i) * (x[1] < self.i + self.th)
            x_idx = (x[0] > self.j) * (x[0] < self.j + self.tw)
            idx = x_idx * y_idx
            x = x[:, idx]
            x[0] -= self.j
            x[1] -= self.i
        else:
            x = x[..., self.i:self.i + self.th, self.j:self.j + self.tw]
        return x


class RandomFlip(object):
    def __init__(self, w, h, flip_h=0.5, flip_v=0.5):
        self.flip_h = random.random() < flip_h
        self.flip_v = random.random() < flip_v
        self.dims = []
        if random.random() < flip_h:
            self.dims.append(-1)
        if random.random() < flip_v:
            self.dims.append(-2)
        self.h, self.w = h - 1, w - 1

    def __call__(self, x, is_evs=False, is_flow=False):
        if is_evs:
            if -1 in self.dims:
                x[0] = self.h - x[0]
            if -2 in self.dims:
                x[1] = self.w - x[1]
        else:
            x = np.flip(x, axis=self.dims)
            if is_flow:
                for d in self.dims:
                    idx = -(d + 1)
                    x[..., idx, :, :] *= -1
        return np.ascontiguousarray(x)


def _grad(func):
    del_func_2d_x = np.diff(func, axis=1)
    del_func_2d_x = np.pad(del_func_2d_x, ((0, 0), (1, 0)), 'edge')
    del_func_2d_y = np.diff(func, axis=0)
    del_func_2d_y = np.pad(del_func_2d_y, ((1, 0), (0, 0)), 'edge')
    return np.concatenate([del_func_2d_x[None], del_func_2d_y[None]], 0)


def norm_fn(evs_matrix):
    with torch.no_grad():
        nonzero_ev = evs_matrix != 0
        num_nonzero = nonzero_ev.sum()
        if num_nonzero > 0:
            mean = evs_matrix.sum() / num_nonzero
            std_dev = torch.sqrt((evs_matrix ** 2).sum() / num_nonzero - mean ** 2)
            mask = nonzero_ev.float()
            evs_matrix = mask * (evs_matrix - mean) / (std_dev + 1.e-8)
    return evs_matrix


def ev_to_voxel(events, B, sensor_size, norm=None):
    xs = torch.from_numpy(events[0, :]).long()
    ys = torch.from_numpy(events[1, :]).long()
    ts = torch.from_numpy(events[2, :]).float()
    ps = torch.from_numpy(events[3, :]).float()
    bins = []
    dt = 1 if ts[-1] - ts[0] == 0 else ts[-1] - ts[0]
    t_norm = (ts - ts[0]) / dt * (B - 1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(B):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        plane = torch.zeros(*sensor_size)
        plane.index_put_((ys, xs), weights, accumulate=True)
        if norm:
            plane = norm_fn(plane)
        bins.append(plane)
    return torch.stack(bins)


class EventDataset(Dataset):
    def __init__(self, args, data_path, train=False):
        self.B = 5
        self.train = train
        self.seq_len = args.seq_len
        self.data_paths = [data_path + '/' + d.name for d in os.scandir(data_path) if d.name.endswith('h5')]
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        self.data_paths.sort(key=alphanum_key)
        self.stop_p = 100 - args.stop_p
        self.noise_mult = args.noise_mult
        self.norm = args.norm

        # extract training sensor size
        data_path = self.data_paths[0]
        with h5py.File(data_path, 'r') as f:
            self.sensor_size = f.attrs['sensor_resolution'][0:2]

        transforms = []
        if train:
            self.noise_paths = [args.noise_path + '/' + d.name for d in os.scandir(args.noise_path) if d.name.endswith('h5')]
            self.noise_paths.sort()

            # extract training noise sensor size
            noise_path = self.noise_paths[0]
            with h5py.File(noise_path, 'r') as f:
                self.noise_size = f.attrs['sensor_resolution'][0:2]

            # setup random augmentation
            transforms.append(RandomCrop(self.sensor_size, args.crop_size))
            transforms.append(RandomFlip(*args.crop_size))

        self.transforms = Compose(transforms)
        self.lenght = len(self.data_paths)

    def __len__(self):
        return self.lenght

    def add_noise(self, ev_data):
        xs = ev_data['xs']
        ys = ev_data['ys']
        ts = ev_data['ts']
        ps = ev_data['ps']
        ts_min = ts[0]
        ts_max = ts[-1]
        ts = (ts - ts_min)
        evs = np.concatenate([xs[None, :], ys[None, :], ts[None, :], ps[None, :]], 0)

        # load a random noise file
        noise_idx = random.randint(0, len(self.noise_paths) - 1)
        noise_path = self.noise_paths[noise_idx]
        with h5py.File(noise_path, 'r') as noise_data:
            xn = np.asarray(noise_data['events/xs'], dtype=np.float32)
            yn = np.asarray(noise_data['events/ys'], dtype=np.float32)
            tn = np.asarray(noise_data['events/ts'], dtype=np.float64)
            pn = np.asarray(noise_data['events/ps'], dtype=np.float32)

        tn /= self.noise_mult
        pn = pn * 2 - 1
        tn = (tn - tn[0]) / 1e6

        # repeat the noise data until it reaches the same timestamp in the train sequence
        while tn[-1] < ts[-1]:
            xn = np.tile(xn, 2)
            yn = np.tile(yn, 2)
            pn = np.tile(pn, 2)
            tn = np.concatenate([tn, tn + tn[-1]])
        cut = np.where(ts[-1] > tn)[0][-1]
        noise = np.concatenate([xn[None, :cut], yn[None, :cut], tn[None, :cut], pn[None, :cut]], 0)

        # crop (spatial) the noise data to the size of the training data (sensor size)
        h, w = int(self.noise_size[0]), int(self.noise_size[1])
        th, tw = int(self.sensor_size[0]), int(self.sensor_size[1])
        ii = random.randint(0, h - th)
        jj = random.randint(0, w - tw)

        y_idx = (noise[1, :] > ii) * (noise[1, :] < ii + th)
        x_idx = (noise[0, :] > jj) * (noise[0, :] < jj + tw)
        idx = x_idx * y_idx

        noise = noise[:, idx]
        noise[1] -= ii
        noise[0] -= jj

        # Concatenate and sort the events and noise
        concat_evs = np.concatenate([evs, noise], 1)
        sorted_evs = concat_evs[:, concat_evs[2, :].argsort()]
        sorted_evs[2] = sorted_evs[2] + ts_min
        return sorted_evs

    def preprocess(self, ev_data, evs, i):
        idx_k = np.searchsorted(evs[2], ev_data['timestamps'][i])
        idx_0 = np.searchsorted(evs[2], ev_data['timestamps'][i - 1])
        evs_ = evs[:, idx_0:idx_k]
        if evs_.shape[1] > 3:
            evs_[2] -= evs_[2, 0]
            vxl = ev_to_voxel(evs_, self.B, self.sensor_size)
            no_flow = False
        else:
            vxl = self.noisy_voxel()
            no_flow = True
        return evs_, vxl, no_flow

    def noisy_voxel(self):
        noise_idx = random.randint(0, len(self.noise_paths) - 1)
        noise_path = self.noise_paths[noise_idx]
        with h5py.File(noise_path, 'r+') as noise_data:
            xn = np.asarray(noise_data['events/xs'], dtype=np.float32)
            yn = np.asarray(noise_data['events/ys'], dtype=np.float32)
            tn = np.asarray(noise_data['events/ts'], dtype=np.float64)
            pn = np.asarray(noise_data['events/ps'], dtype=np.float32)

        pn = pn * 2 - 1
        tn = (tn - tn[0]) / 1e6
        min_sample = min(10000, xn.shape[0] - 1)
        cut = random.randint(min_sample, xn.shape[0] - 1)
        evs_ = np.concatenate([xn[None, :cut], yn[None, :cut], tn[None, :cut], pn[None, :cut]], 0)

        h, w = int(self.noise_size[0]), int(self.noise_size[1])
        th, tw = int(self.sensor_size[0]), int(self.sensor_size[1])
        ii = random.randint(0, h - th)
        jj = random.randint(0, w - tw)

        y_idx = (evs_[1, :] > ii) * (evs_[1, :] < ii + th)
        x_idx = (evs_[0, :] > jj) * (evs_[0, :] < jj + tw)
        idx = x_idx * y_idx

        evs_ = evs_[:, idx]
        evs_[1] -= ii
        evs_[0] -= jj

        vxl = ev_to_voxel(evs_, self.B, self.sensor_size)
        return vxl[None]

    def count_fn(self, evs, grads, flow):
        evs = np.asarray(evs, dtype=np.int32)
        xs = evs[0, :].copy().tolist()
        ys = evs[1, :].copy().tolist()
        ps = evs[3, :].copy()
        g_val = grads[:, ys, xs]
        f_val = flow[:, ys, xs]
        ev_grad = np.concatenate([evs[0:1, :], evs[1:2, :],
                                  g_val[0:1, :], g_val[1:2, :],
                                  f_val[0:1, :], f_val[1:2, :]], 0)
        return np.float32(ev_grad)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        with h5py.File(data_path, 'r') as f:
            if self.seq_len:
                seq_len = min(self.seq_len, f.attrs['num_imgs'])
            else:
                seq_len = f.attrs['num_imgs']

            event_idx = []
            timestamps = []
            imgs = []
            flw = []
            for k, kk in zip(f['images'].keys(), f['flow'].keys()):
                event_idx.append(f['images'][k].attrs['event_idx'])
                imgs.append(np.asarray(f['images'][k]))
                flw.append(np.asarray(f['flow'][kk]))
                assert f['images'][k].attrs['timestamp'] == f['flow'][kk].attrs['timestamp']
                timestamps.append(f['images'][k].attrs['timestamp'])
                if len(imgs) >= seq_len:
                    break
            ev_data = {
                'xs': np.asarray(f['events/xs'], dtype=np.int16),
                'ys': np.asarray(f['events/ys'], dtype=np.int16),
                'ts': np.asarray(f['events/ts'], dtype=np.float64),
                'ps': np.asarray(f['events/ps'], dtype=np.int8) * 2 - 1,
                'num_imgs': seq_len,
                'images': imgs,
                'event_idx': event_idx,
                'timestamps': timestamps,
                'flow': flw,
            }

        # add noise if training
        if self.train:
            evs = self.add_noise(ev_data)
        else:
            xs = ev_data['xs']
            ys = ev_data['ys']
            ts = ev_data['ts']
            ps = ev_data['ps']
            evs = np.concatenate([xs[None, :], ys[None, :], ts[None, :], ps[None, :]], 0)

        # stopping probability
        if random.randint(0, 99) > self.stop_p and self.train and seq_len:
            if seq_len >= self.seq_len:
                p1 = int(random.randint(20, 40) * seq_len / 100)
                p2 = p1 + int(random.randint(13, 33) * seq_len / 100)
                noise_len = p1
                p3 = seq_len
                seq_list = list(range(p1, p2, 1)) + list(range(p2, p3, 1))
            else:
                seq_list = list(range(1, seq_len, 1))
                p1 = 1
                p2 = seq_len
                p3 = p2
                noise_len = self.seq_len - (seq_len - p1)
            stoping = True
        else:
            seq_list = list(range(0, seq_len, 1))
            stoping = False

        imgs = np.zeros([seq_len, *self.sensor_size], dtype=np.float32)
        grads_full = np.zeros([seq_len, 2, *self.sensor_size], dtype=np.float32)
        flws = np.zeros([seq_len, 2, *self.sensor_size], dtype=np.float32)
        voxel = np.zeros([seq_len, self.B, *self.sensor_size], dtype=np.float32)

        evs_data, evs_grad = {}, {}
        for ii, i in enumerate(seq_list):
            evs_, vxl, no_flow = self.preprocess(ev_data, evs, i)
            if no_flow:
                flws[ii] = np.zeros([2, *self.sensor_size], dtype=np.float32)
            else:
                flws[ii] = ev_data['flow'][i] * (evs_[2, -1] - evs_[2, 0])

            evs_data[ii] = self.transforms(evs_, is_evs=True, is_flow=False).T
            imgs[ii] = ev_data['images'][i]
            grads_full[ii] = _grad(imgs[ii])
            ev_grads = self.count_fn(evs_, grads_full[i], flws[i])
            evs_grad[ii] = self.transforms(ev_grads, is_evs=True, is_flow=False).T
            voxel[ii] = vxl

        if stoping:
            stop_img = np.tile(imgs[p2 - p1 - 1][None], [noise_len, 1, 1])
            imgs = np.concatenate([imgs[0:p2 - p1], stop_img, imgs[p2 - p1:p3 - p1]], 0)
            stop_vxl = torch.cat([self.noisy_voxel() for i in range(noise_len)], 0)
            voxel = np.concatenate([voxel[0:p2 - p1], stop_vxl, voxel[p2 - p1:p3 - p1]], 0)
            stop_grad = np.tile(grads_full[p2 - p1 - 1][None], [noise_len, 1, 1, 1])
            grads_full = np.concatenate([grads_full[0:p2 - p1], stop_grad, grads_full[p2 - p1:p3 - p1]], 0)
            stop_flow = np.zeros([noise_len, 2, *self.sensor_size], dtype=np.float32)
            flws = np.concatenate([flws[0:p2 - p1], stop_flow, flws[p2 - p1:p3 - p1]], 0)
        else:
            if seq_len < self.seq_len:
                noise_len = self.seq_len - seq_len
                stop_img = np.tile(imgs[-1][None], [noise_len, 1, 1])
                imgs = np.concatenate([imgs, stop_img], 0)
                stop_vxl = torch.cat([self.noisy_voxel() for i in range(noise_len)], 0)
                voxel = np.concatenate([voxel, stop_vxl], 0)
                stop_grad = np.tile(grads_full[-1][None], [noise_len, 1, 1, 1])
                grads_full = np.concatenate([grads_full, stop_grad], 0)
                stop_flow = np.zeros([noise_len, 2, *self.sensor_size], dtype=np.float32)
                flws = np.concatenate([flws, stop_flow], 0)

        imgs = self.transforms(imgs, is_evs=False, is_flow=False)
        voxel = self.transforms(voxel, is_evs=False, is_flow=False)
        grads_full = self.transforms(grads_full, is_evs=False, is_flow=False)
        flws = self.transforms(flws, is_evs=False, is_flow=True)
        imgs = np.tile(imgs[:, None], [1, 3, 1, 1])

        assert imgs.shape[0] == voxel.shape[0] == grads_full.shape[0] == flws.shape[0]
        seq_len = imgs.shape[0]

        return {
            'images': imgs / 255.0,
            'voxel': voxel,
            'grads': grads_full / 255.0,
            'flow': flws,
            'len': seq_len,
            'res': self.sensor_size,
            'data_path': data_path
        }
