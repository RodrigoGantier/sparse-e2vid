import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import numpy as np
import random
import os
from tqdm import trange
from tqdm import tqdm

from tools.loss_fn import f_Loss
from tools.loss_fn import frankotchellappa
from models.small_e2v3_SubMsparse5 import e2v
from dataset import EventDataset


def seed_everything(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')


def train(model, data, loss_fn, optm, scheduler, e, device):
    epoch_loss = 0
    epoch_lpips = 0

    for ii, batch in enumerate(tqdm(data)):
        seq_len = batch['len'][0]
        lpips_loss = 0
        mse_loss = 0
        tc_loss = 0
        total_loss = 0
        model.reset_states()
        optm.zero_grad()

        loss_fn.last_t = batch['images'][:, 0].to(device)

        for i in range(1, seq_len, 1):
            voxel = batch['voxel'][:, i].to(device)
            grad = batch['grads'][:, i].to(device)
            img = batch['images'][:, i].to(device)
            flow = batch['flow'][:, i].to(device)

            output = model(voxel)
            lpips, mse, tc = loss_fn(output, grad, img, flow, normalize=True)
            with torch.no_grad():
                lpips_loss = (lpips + ((i - 1) * lpips_loss)) / i
                mse_loss = (mse + ((i - 1) * mse_loss)) / i
                tc_loss = (tc + ((i - 1) * tc_loss)) / i
            total_loss = (((lpips + mse + tc) / 3) + ((i - 1) * total_loss)) / i

        total_loss.backward()
        optm.step()
        scheduler.step()

        global_step = (len(data) * e) + ii

        output = frankotchellappa(output)

        with torch.no_grad():
            epoch_loss = (total_loss.item() + (ii * epoch_loss)) / (ii + 1)
            epoch_lpips = (lpips_loss.item() + (ii * epoch_lpips)) / (ii + 1)

    return epoch_loss


def main(args):
    seed_everything(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    tr_data = EventDataset(args, args.tr_path, train=True)
    tr = DataLoader(tr_data, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    model = e2v(args)
    model.to(args.device)
    model.train()

    optm = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        amsgrad=True,
        weight_decay=1e-3)
    scheduler = OneCycleLR(
        optm, max_lr=args.lr,
        div_factor=2.0,
        total_steps=len(tr) * args.epochs + 1)

    loss_fn = f_Loss(args.crop_size, device=args.device, net='vgg')
    best = 1000
    for e in trange(args.epochs):
        loss = train(model, tr, loss_fn, optm, scheduler, e, args.device)

        torch.save({'state_dict': model.state_dict(),
                    'epoch': e,
                    'optimizer_state_dict': optm.state_dict()},
                   os.path.join(args.save_path, 'last.pth'))

        if loss < best:
            torch.save(
                {'state_dict': model.state_dict(),
                 'epoch': e,
                 'optimizer_state_dict': optm.state_dict()},
                os.path.join(args.save_path, 'best.pth'))
            best = loss

        print(f"Epoch {e}: loss={loss:.4f}, best={best:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gradient prediction - Training')
    parser.add_argument('--tr_path', default='./data/train')
    parser.add_argument('--noise_path', default='./data/noise')
    parser.add_argument('--save_path', default='./checkpoints/training')
    parser.add_argument('--device', default='cuda:0', type=str)

    # net parameters
    parser.add_argument('--in_chans', default=5)
    parser.add_argument('--out_chans', default=2)
    parser.add_argument('--embed_dim', default=16)
    parser.add_argument('--kernel_size', default=3)
    parser.add_argument('--num_bins', default=5)

    # train parameters
    parser.add_argument('--norm', default=False, type=bool)
    parser.add_argument('--crop_size', default=(180, 180))
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--bs', default=1, type=int, help='Batch size')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seq_len', default=25, help='length of the sequence')
    parser.add_argument('--stop_p', default=80)
    parser.add_argument('--noise_mult', default=100)

    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    main(args)
