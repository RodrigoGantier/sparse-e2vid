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
from tqdm import trange

# ---------------------------------------------------------------------------
# Environment-level optimizations
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# TF32 gives free speed-up on Ampere+ GPUs with negligible accuracy loss.
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False   # benchmark=True needs this False
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
        output = F.pad(output, (0, output.shape[3] - 1, 0, output.shape[2] - 1), mode='reflect')
        del_f_del_x, del_f_del_y = output[:, 0:1], output[:, 1:2]

        x_fft = fft2(del_f_del_x)
        y_fft = fft2(del_f_del_y)

        numerator = -1j * self.wx * x_fft - 1j * self.wy * y_fft
        denominator = (self.wx) ** 2 + (self.wy) ** 2 + torch.finfo(float).eps

        res = ifft2(numerator / denominator)
        res -= torch.mean(torch.real(res))

        res = res[:, :, :-h + self.pd, :-w + self.pd]
        res = torch.real(res) * (-1)

        return res


class IntensityRescaler:
    def __init__(self, args, int):
        self.auto_hdr = True
        self.intensity_bounds = []
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
            self.intensity_bounds.pop(0)

        self.intensity_bounds.append((Imin, Imax))
        self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
        self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img[0].mean(0).byte()

        return img.detach().cpu().numpy()


class OptimizedVoxelDataset:
    """
    VoxelDataset with GPU-side event voxelization to avoid CPU->GPU copies.
    """
    def __init__(self, args):
        self.B = args.num_bins
        self.data_path = args.data_path
        self.device = args.device
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

    def __exit__(self):
        self.data.close()

    def process_evs(self, i):
        voxels = []
        last_idx, next_idx = 0, self.num_evs

        i0, i1 = self.idxs[i]
        max_len = i1 - i0

        while max_len > next_idx:
            xs, ys, ps, ts = self.read_evs(last_idx, next_idx, i)
            voxels.append(self.events_to_voxel_grid_gpu(xs, ys, ts, ps))
            last_idx = next_idx
            next_idx = np.clip(next_idx + self.num_evs, 0, max_len + 1)

        if max_len <= next_idx:
            if next_idx - last_idx < 3 or max_len < 3:
                num_evs = torch.randint(10, 100, (1,), device=self.device).item()
                xs = torch.randint(0, self.sensor_size[1], (num_evs,), device=self.device)
                ys = torch.randint(0, self.sensor_size[0], (num_evs,), device=self.device)
                ts = torch.rand(num_evs, device=self.device) * 1000.0
                ps = torch.randint(0, 2, (num_evs,), device=self.device).float()
                ps[ps == 0] = -1.0

                voxels.append(self.events_to_voxel_grid_gpu(xs, ys, ts, ps))
            else:
                xs, ys, ps, ts = self.read_evs(last_idx, max_len, i)
                voxels.append(self.events_to_voxel_grid_gpu(xs, ys, ts, ps))

        return voxels

    def read_evs(self, last_idx, next_idx, i):
        i0, i1 = self.idxs[i]
        xs = self.data['events/xs'][i0:i1]
        ys = self.data['events/ys'][i0:i1]
        ps = self.data['events/ps'][i0:i1]
        ts = self.data['events/ts'][i0:i1]

        # Directly create tensors on target device to avoid H2D copy later
        xs = torch.from_numpy(xs[last_idx:next_idx]).long().to(self.device)
        ys = torch.from_numpy(ys[last_idx:next_idx]).long().to(self.device)
        ps = torch.from_numpy(ps[last_idx:next_idx]).float().to(self.device)
        ts = ts[last_idx:next_idx]
        ts -= ts[0]
        ts = torch.from_numpy(ts).float().to(self.device)
        ts /= 1e6
        ps[ps == 0] = -1.0

        return xs, ys, ps, ts

    def events_to_voxel_grid_gpu(self, xs, ys, ts, ps):
        """Build voxel grid entirely on GPU."""
        with torch.no_grad():
            H, W = self.sensor_size
            bins = []
            dt = ts[-1] - ts[0]
            if dt == 0:
                dt = 1e-6
            t_norm = (ts - ts[0]) / dt * (self.B - 1)
            zeros = torch.zeros_like(t_norm)

            for bi in range(self.B):
                bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
                weights = ps * bilinear_weights
                plane = torch.zeros(H, W, device=self.device)
                plane.index_put_((ys, xs), weights, accumulate=True)
                bins.append(plane)

        bins = torch.stack(bins)
        if self.filter_hot_events:
            bins = self.hot_mask_gpu(bins, 0.9)

        if self.norm:
            bins = self.norm_fn_gpu(bins)

        return bins

    def norm_fn_gpu(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)
        return events

    def hot_mask_gpu(self, events, thr):
        with torch.no_grad():
            evs = events.sum(0)

            evs_pos = torch.where(evs > 0, 1.0, 0.0) * evs
            num_nonzeros = (evs_pos != 0).sum()
            if num_nonzeros > 0:
                mean = evs_pos.sum() / num_nonzeros
                stddev = torch.sqrt((evs_pos ** 2).sum() / num_nonzeros - mean ** 2)
                pos_mask = torch.where(evs_pos > (mean + stddev * thr), 0.0, 1.0)
            else:
                pos_mask = torch.ones_like(evs)

            evs_neg = torch.where(evs < 0, 1.0, 0.0) * evs
            num_nonzeros = (evs_neg != 0).sum()
            if num_nonzeros > 0:
                mean = evs_neg.sum() / num_nonzeros
                stddev = torch.sqrt((evs_neg ** 2).sum() / num_nonzeros - mean ** 2)
                neg_mask = torch.where(evs_neg < (mean - stddev * thr), 0.0, 1.0)
            else:
                neg_mask = torch.ones_like(evs)

        return events * neg_mask * pos_mask


def model_load(args, model):
    ckpt = torch.load(args.save_path, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
    return model


@torch.inference_mode()
def main(args):
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OptimizedVoxelDataset(args)
    model = e2v(args).to(args.device)
    model = model_load(args, model)
    model = model.float().to(args.device)
    model.reset_states()
    model.eval()
    args.crop_size = dataset.sensor_size.tolist()
    out_rescal = IntensityRescaler(args, True)

    # -----------------------------------------------------------------------
    # 1. torch.compile (PyTorch 2.0+) - best free speed-up for dense ops
    # -----------------------------------------------------------------------
    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode=args.compile_mode)
                print(f"[OPT] torch.compile enabled (mode={args.compile_mode})")
            except Exception as e:
                print(f"[OPT] torch.compile failed: {e}")
        else:
            print("[OPT] torch.compile not available (needs PyTorch >= 2.0)")

    # -----------------------------------------------------------------------
    # 2. Automatic Mixed Precision (AMP) / FP16
    # -----------------------------------------------------------------------
    amp_enabled = False
    if args.fp16:
        print("[OPT] FP16 (half precision) enabled")
        model = model.half()
        amp_enabled = True

    # -----------------------------------------------------------------------
    # 3. Warm-up with identical pipeline to the real run
    # -----------------------------------------------------------------------
    print("[OPT] Warming up GPU...")
    warmup_dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.zeros(1, args.num_bins, dataset.h, dataset.w,
                        device=args.device, dtype=warmup_dtype)
    for _ in range(10):
        if args.fp16:
            with torch.cuda.amp.autocast():
                _ = model(dummy)
        else:
            _ = model(dummy)
    model.reset_states()
    torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Benchmark / inference loop
    # -----------------------------------------------------------------------
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    num_frames = 0

    print(f"[OPT] Running inference on {len(dataset.imgs)} frames...")
    for i in trange(len(dataset.imgs), desc='Inference'):
        voxels = dataset.process_evs(i)

        start_event.record()
        for voxel in voxels:
            voxel = voxel[None].to(args.device)
            if args.fp16:
                voxel = voxel.half()

            if amp_enabled:
                with torch.cuda.amp.autocast():
                    output = model(voxel)
            else:
                output = model(voxel)

            pred = out_rescal(output)
        end_event.record()

        if args.benchmark_sync:
            torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
        num_frames += 1

        out_path = os.path.join(args.output_dir, f'frame_{i:04d}.png')
        cv2.imwrite(out_path, pred)

    avg_time = total_time / max(num_frames, 1)
    fps = 1000.0 / avg_time
    print(f'\n[RESULT] Total frames : {num_frames}')
    print(f'[RESULT] Avg latency  : {avg_time:.2f} ms')
    print(f'[RESULT] Effective FPS: {fps:.2f}')
    print(f'[RESULT] Output saved to: {args.output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sparse E2VID - Optimized Inference')
    parser.add_argument('--data_path', default='./data/bike_bay_hdr.h5')
    parser.add_argument('--save_path', default='./checkpoints/last_0150.pth', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--output_dir', default='./output_optim', type=str)

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

    # -----------------------------------------------------------------------
    # Optimization flags
    # -----------------------------------------------------------------------
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in FP16 (half precision) for ~2x throughput on modern GPUs')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile (PyTorch 2.0+)')
    parser.add_argument('--compile_mode', default='reduce-overhead', type=str,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--benchmark_sync', action='store_true',
                        help='Enable cuda.synchronize() every frame for precise timing (slightly slower)')

    args = parser.parse_args()
    main(args)
