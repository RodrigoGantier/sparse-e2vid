import torch
import os
import argparse
import numpy as np
import cv2
import random
import h5py
from torch.fft import fft2, ifft2, fftfreq
import torch.nn.functional as F

from models.small_e2v3_SubMsparse5 import e2v
from tools.timers import CudaTimer_ma
from collections import deque
from tools.metrics import metrics_Fn
from tqdm import trange
import pandas as pd


def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class fK_fn:
    def __init__(self, img_size, device, padding):
        self.pd = (padding * 2) + 1
        pd = 2 * (padding * 2) + 1
        NN, MM = img_size[0] * 2 - pd, img_size[1] * 2 - pd

        wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                             fftfreq(NN) * 2 * np.pi, indexing='xy')
        self.wx = torch.from_numpy(wx).to(device)
        self.wy = torch.from_numpy(wy).to(device)

    def __call__(self, output):
        h, w = output.shape[-2:]
        output = torch.nn.functional.pad(output, (0, output.shape[3]-1, 0, output.shape[2]-1), mode='reflect')
        del_f_del_x, del_f_del_y = output[:, 0:1], output[:, 1:2]

        x_fft = fft2(del_f_del_x)
        y_fft = fft2(del_f_del_y)

        numerator = -1j * self.wx * x_fft - 1j * self.wy * y_fft
        denominator = (self.wx) ** 2 + (self.wy) ** 2 + torch.finfo(float).eps

        res = ifft2(numerator / denominator)
        res -= torch.mean(torch.real(res))

        res = res[:, :, :-h+self.pd, :-w+self.pd]
        res = torch.real(res)*(-1)

        return res


class IntensityRescaler:
    def __init__(self, args, int):
        self.auto_hdr = True
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = 10
        self.Imin = 0.0
        self.Imax = 1.0
        self.pd = 0
        self.integrate = fK_fn(args.crop_size, args.device, self.pd)
        self.int = int

    def __call__(self, img):
        img = img.float()
        if self.pd > 0:
            img = img[:, :, self.pd:-self.pd, self.pd:-self.pd]
        if self.int:
            img = self.integrate(img)

        Imin = torch.min(img).item()
        Imax = torch.max(img).item()

        if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
            self.intensity_bounds.popleft()

        self.intensity_bounds.append((Imin, Imax))
        self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
        self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img[0].mean(0).byte()

        return img.detach().cpu().numpy()


class VoxelDataset:
    def __init__(self, args):
        self.B = args.num_bins
        self.data_path = args.data_path
        self.idxs = {}
        self.imgs = []
        self.data = h5py.File(self.data_path, 'r')

        self.sensor_size = self.data.attrs['sensor_resolution'][0:2]
        i0 = 0
        for i, k in enumerate(self.data['images'].keys()):
            self.imgs.append(np.fliplr(np.asarray(self.data['images'][k])))
            i1 = self.data['images'][k].attrs['event_idx']
            self.idxs[i] = [i0, i1]
            i0 = i1.copy()
        self.norm = args.norm_input

        self.h, self.w = self.sensor_size
        self.filter_hot_events = args.filter_hot_events
        self.num_evs = int(self.h * self.w * args.ev_rate)

    def __exit__(self, ):
        self.data.close()

    def process_evs(self, i):
        voxels = []
        last_idx, next_idx = 0, self.num_evs

        i0, i1 = self.idxs[i]
        max_len = i1 - i0

        while max_len > next_idx:
            xs, ys, ps, ts = self.read_evs(last_idx, next_idx, i)
            voxels.append(self.events_to_voxel_grid_pytorch(xs, ys, ts, ps))
            last_idx = next_idx
            next_idx = np.clip(next_idx + self.num_evs, 0, max_len + 1)

        if max_len <= next_idx:
            if next_idx-last_idx < 3 or max_len < 3:
                num_evs = torch.FloatTensor(1, 1).uniform_(10, 100).long().item()
                xs = torch.FloatTensor(1, num_evs).uniform_(0, self.sensor_size[1]).reshape(-1).long()
                ys = torch.FloatTensor(1, num_evs).uniform_(0, self.sensor_size[0]).reshape(-1).long()
                ts = torch.FloatTensor(1, num_evs).uniform_(0, 1000).reshape(-1).float()
                ps = torch.FloatTensor(1, num_evs).uniform_(0, 2).reshape(-1).float()
                ps[ps == 0] = -1

                voxels.append(self.events_to_voxel_grid_pytorch(xs, ys, ts, ps))
            else:
                xs, ys, ps, ts = self.read_evs(last_idx, max_len, i)
                voxels.append(self.events_to_voxel_grid_pytorch(xs, ys, ts, ps))

        return voxels

    def read_evs(self, last_idx, next_idx, i):
        i0, i1 = self.idxs[i]
        xs = self.data['events/xs'][i0:i1]
        ys = self.data['events/ys'][i0:i1]
        ps = self.data['events/ps'][i0:i1]
        ts = self.data['events/ts'][i0:i1]

        xs = torch.from_numpy(xs[last_idx:next_idx]).long()
        ys = torch.from_numpy(ys[last_idx:next_idx]).long()
        ps = torch.from_numpy(ps[last_idx:next_idx]).float()
        ts = ts[last_idx:next_idx]
        ts -= ts[0]
        ts = torch.from_numpy(ts).float()
        ts /= 1e6
        ps[ps == 0] = -1

        return xs, ys, ps, ts

    def events_to_voxel_grid_pytorch(self, xs, ys, ts, ps):
        with torch.no_grad():
            bins = []
            dt = ts[-1]-ts[0]
            t_norm = (ts-ts[0])/dt*(self.B-1)
            zeros = torch.zeros(t_norm.size())
            for bi in range(self.B):
                bilinear_weights = torch.max(zeros, 1.0-torch.abs(t_norm-bi))
                weights = ps*bilinear_weights
                plane = torch.zeros(*self.sensor_size)
                plane.index_put_((ys, xs), weights, accumulate=True)
                bins.append(plane)

        bins = torch.stack(bins)
        if self.filter_hot_events:
            bins = self.hotMask(bins, 0.9)

        if self.norm:
            bins = self.norm_fn(bins)

        return bins

    def norm_fn(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)
        return events

    def hotMask(self, events, thr):
        with torch.no_grad():
            evs = events.sum(0)

            evs_pos = torch.where(evs > 0, 1, 0) * evs
            num_nonzeros = (evs_pos != 0).sum()
            if num_nonzeros > 0:
                mean = evs_pos.sum() / num_nonzeros
                stddev = torch.sqrt((evs_pos ** 2).sum() / num_nonzeros - mean ** 2)
                pos_mask = torch.where(evs_pos > (mean + stddev * thr), 0.0, 1.0)
            else:
                pos_mask = torch.ones_like(evs)

            evs_neg = torch.where(evs < 0, 1, 0) * evs
            num_nonzeros = (evs_neg != 0).sum()
            if num_nonzeros > 0:
                mean = evs_neg.sum() / num_nonzeros
                stddev = torch.sqrt((evs_neg ** 2).sum() / num_nonzeros - mean ** 2)
                neg_mask = torch.where(evs_neg < (mean - stddev * thr), 0.0, 1.0)
            else:
                neg_mask = torch.ones_like(evs)

        return events * neg_mask * pos_mask

    def clip_fn(self, events):
        with torch.no_grad():
            nonzero_ev = (events > 0)
            pos_evs = events * nonzero_ev
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = pos_evs.sum() / num_nonzeros
                stddev = torch.sqrt((pos_evs ** 2).sum() / num_nonzeros - mean ** 2)
                events = torch.clip(events, events.min(), mean + stddev * 0.2)

            nonzero_ev = (events < 0)
            neg_evs = events * nonzero_ev
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = neg_evs.sum() / num_nonzeros
                stddev = torch.sqrt((neg_evs ** 2).sum() / num_nonzeros - mean ** 2)
                events = torch.clip(events, mean - stddev * 0.2, events.max())

        return events


def model_load(args, model):
    ckpt = torch.load(args.save_path)
    model.load_state_dict(ckpt['state_dict'])
    return model


@torch.no_grad()
def evaluate_dataset(args, data_path):
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(data_path)}")
    print(f"{'='*60}")

    args.data_path = data_path
    dataset = VoxelDataset(args)
    model = e2v(args).to(args.device)
    model = model_load(args, model)
    model = model.float().to(args.device)
    model.reset_states()
    model.eval()
    args.crop_size = dataset.sensor_size.tolist()
    out_rescal = IntensityRescaler(args, True)
    img_rescal = IntensityRescaler(args, False)
    time_profile = CudaTimer_ma('inference')
    crop_size = (dataset.h, dataset.w)
    metrics = metrics_Fn(crop_size, args.device)
    ssim = []
    mse = []
    tc = []
    lpips = []

    for i in trange(len(dataset.imgs), desc=os.path.basename(data_path)):
        voxels = dataset.process_evs(i)
        img1 = dataset.imgs[i][:, ::-1].copy()

        for voxel in voxels:
            voxel = voxel[None].to(args.device).float()
            with time_profile:
                output = model(voxel)
                pred1 = out_rescal(output)

        img1 = torch.from_numpy(img1[None, None]).float()
        img1 = img_rescal(img1/255.0)
        pred1 = cv2.resize(pred1, (img1.shape[1], img1.shape[0]))
        img1 = torch.from_numpy(img1[None, None]).float()
        img1 = img1.repeat(1, 3, 1, 1).to(args.device)

        pred1 = torch.from_numpy(pred1[None, None]).float()
        pred1 = pred1.repeat(1, 3, 1, 1).to(args.device)

        if i == 0:
            metrics.img0 = img1.float()
            metrics.pred0 = pred1.float()
            continue

        tc.append(metrics.temp_loss(pred1, img1))
        ssim.append(metrics.ssim_metric(pred1, img1))
        mse.append(metrics.mse_metric(pred1, img1))
        lpips.append(metrics.lpips_fn(pred1, img1))

    print(f'time: {time_profile.av_time}')
    print(f'time consistency error: {np.mean(tc)}')
    print(f'LPIPS error: {np.mean(lpips)}')
    print(f'SSIM error: {np.mean(ssim)}')
    print(f'MSE error: {np.mean(mse)}')

    return {
        'dataset': os.path.basename(data_path),
        'time': time_profile.av_time,
        'tc': np.mean(tc),
        'lpips': np.mean(lpips),
        'ssim': np.mean(ssim),
        'mse': np.mean(mse)
    }


@torch.no_grad()
def main(args):
    seed_everything(args.seed)

    test_datasets = [
        './data/bike_bay_hdr.h5',
        './data/boxes.h5',
        './data/desk.h5',
        './data/desk_fast.h5',
        './data/desk_hand_only.h5',
        './data/desk_slow.h5',
        './data/engineering_posters.h5',
        './data/high_texture_plants.h5',
        './data/poster_pillar_1.h5',
        './data/poster_pillar_2.h5',
        './data/reflective_materials.h5',
        './data/slow_and_fast_desk.h5',
        './data/slow_hand.h5',
        './data/still_life.h5',
    ]

    results = []
    for data_path in test_datasets:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping...")
            continue
        result = evaluate_dataset(args, data_path)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(results)
    if not df.empty:
        # Reorder columns for readability
        df = df[['dataset', 'tc', 'lpips', 'ssim', 'mse', 'time']]
        df.columns = ['Dataset', 'TC', 'LPIPS', 'SSIM', 'MSE', 'Time (ms)']

        # Print as a nicely formatted table
        print(df.to_string(index=False))

        # Also save to CSV for later use
        csv_path = './test_results.csv'
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\nResults saved to: {csv_path}")
    else:
        print("No results to display.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gradient prediction - Test all datasets')
    parser.add_argument('--save_path', default='./checkpoints/last_0150.pth', type=str)
    # parser.add_argument('--save_path', default='./checkpoints/best.pth', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)

    # net parameters
    parser.add_argument('--embed_dim', default=16)
    parser.add_argument('--num_bins', default=5)
    parser.add_argument('--in_chans', default=5)
    parser.add_argument('--out_chans', default=2)
    parser.add_argument('--kernel_size', default=3)

    # train parameters
    parser.add_argument('--norm_input', default=False, type=bool)
    parser.add_argument('--ev_rate', default=0.1, type=float)
    parser.add_argument('--filter_hot_events', type=bool, default=False)

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    main(args)
