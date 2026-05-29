import torch
import numpy as np
import torch.nn.functional as F
import lpips
from tools import loss as flow_loss
from torch.fft import fft2, ifft2, fftfreq


class combined_perceptual_loss():
    def __init__(self, weight=1.0, device='cpu'):
        self.loss = perceptual_loss(weight=1.0, device=device)
        self.weight = weight

    def __call__(self, pred_img, pred_flow, target_img, target_flow):
        pred = torch.cat([pred_img, pred_flow], dim=1)
        target = torch.cat([target_img, target_flow], dim=1)
        dist = self.loss(pred, target, normalize=False)
        return dist * self.weight


class warping_flow_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = flow_loss.warping_flow_loss
        self.weight = weight
        self.L0 = L0
        self.default_return = None

    def __call__(self, i, image1, flow):
        loss = self.default_return if i < self.L0 else self.weight * self.loss(
                self.image0, image1, -flow)
        self.image0 = image1
        return loss


class voxel_warp_flow_loss():
    def __init__(self, weight=1.0):
        self.loss = flow_loss.voxel_warping_flow_loss
        self.weight = weight

    def __call__(self, voxel, displacement, output_images=False):
        loss = self.loss(voxel, displacement, output_images)
        if output_images:
            loss = (self.weight * loss[0], loss[1])
        else:
            loss *= self.weight
        return loss


class flow_perceptual_loss():
    def __init__(self, weight=1.0, device='cpu'):
        self.loss = perceptual_loss(weight=1.0, device=device)
        self.weight = weight

    def __call__(self, pred, target):
        dist_x = self.loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], normalize=False)
        dist_y = self.loss(pred[:, 1:2, :, :], target[:, 1:2, :, :], normalize=False)
        return (dist_x + dist_y) / 2 * self.weight


class flow_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', device='cpu'):
        self.model = lpips.LPIPS(net=net).to(device)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class temporal_consistency_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = flow_loss.temporal_consistency_loss
        self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             -flow, output_images=output_images)
            if output_images:
                loss = (self.weight * loss[0], loss[1])
            else:
                loss *= self.weight
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss


def _reflec_pad_grad_fields(del_func_x, del_func_y):
    del_func_x_c1 = torch.cat((del_func_x, del_func_x[::-1, :]), axis=0)
    del_func_x_c2 = torch.cat((-del_func_x[:, ::-1], -del_func_x[::-1, ::-1]), axis=0)
    del_func_x = torch.cat((del_func_x_c1, del_func_x_c2), axis=1)
    del_func_y_c1 = torch.cat((del_func_y, -del_func_y[::-1, :]), axis=0)
    del_func_y_c2 = torch.cat((del_func_y[:, ::-1], -del_func_y[::-1, ::-1]), axis=0)
    del_func_y = torch.cat((del_func_y_c1, del_func_y_c2), axis=1)
    return del_func_x, del_func_y


def _one_forth_of_array(array):
    array, _ = torch.tensor_split(array, 2, dim=0)
    array = torch.tensor_split(array, 2, axis=1)[0]
    return array


def frankotchellappa(output):
    h, w = output.shape[-2:]
    output = torch.nn.functional.pad(output, (0, output.shape[-1]-1, 0, output.shape[-2]-1), mode='reflect')
    del_f_del_x, del_f_del_y = output[:, 0:1], output[:, 1:2]

    NN, MM = del_f_del_x.shape[-2:]
    wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                         fftfreq(NN) * 2 * np.pi, indexing='xy')

    wx = torch.from_numpy(wx).to(output.device)
    wy = torch.from_numpy(wy).to(output.device)

    x_fft = fft2(del_f_del_x)
    y_fft = fft2(del_f_del_y)

    numerator = -1j * wx * x_fft - 1j * wy * y_fft
    denominator = (wx) ** 2 + (wy) ** 2 + torch.finfo(float).eps

    res = ifft2(numerator / denominator)
    res -= torch.mean(torch.real(res))

    res = res[..., :-h+1, :-w+1]
    res = torch.real(res)
    B, C, H, W = res.size()
    return res.expand(B, 3, H, W)


class denseLoss_2():
    def __init__(self, sensor_size, device, net='alex', alpha=50):
        self.model = lpips.LPIPS(net=net).to(device)
        self.mse_loss = F.mse_loss
        self.l1_loss = F.l1_loss
        self.alpha = alpha
        self.t_width, self.t_height = sensor_size
        self.last_p = None
        self.last_t = None

    def __call__(self, pred, trgt, flow, normalize=True):
        lpips_val = self.model(pred, trgt, normalize=normalize).mean()
        mse = self.mse_loss(pred, trgt) * 0.2
        return lpips_val, mse


class denseLoss():
    def __init__(self, sensor_size, device, net='alex', alpha=50):
        self.model = lpips.LPIPS(net=net).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.alpha = alpha
        self.t_width, self.t_height = sensor_size

        xx, yy = torch.meshgrid(torch.arange(self.t_width), torch.arange(self.t_height))
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        self.xx, self.yy = xx, yy

        self.last_p = None
        self.last_t = None

    def __call__(self, pred, trgt, flow, normalize=True):
        flow_x = flow[:, 0, :, :]
        flow_y = flow[:, 1, :, :]

        warping_grid_x = self.xx.to(pred.device) - flow_x
        warping_grid_y = self.yy.to(pred.device) - flow_y

        warping_grid_x = (2 * warping_grid_x / (self.t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (self.t_height - 1)) - 1
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)

        image1_warped_to0 = F.grid_sample(trgt, warping_grid, align_corners=True)
        prod1_warped_to0 = F.grid_sample(pred, warping_grid, align_corners=True)

        visibility_mask = torch.exp(-self.alpha * (self.last_t - image1_warped_to0) ** 2)
        tc_loss = visibility_mask * torch.abs(self.last_t - prod1_warped_to0) \
             / (torch.abs(self.last_t) + torch.abs(prod1_warped_to0) + 1e-5)

        lpips_val = self.model(pred, trgt, normalize=normalize).mean().to(pred.device)
        mse = torch.mul(self.mse_loss(pred, trgt), 0.2).to(pred.device)

        return lpips_val, mse, tc_loss.mean().to(pred.device)


class f_Loss():
    def __init__(self, sensor_size, device, net='alex', alpha=50):
        self.model = lpips.LPIPS(net=net).to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.alpha = alpha
        self.t_width, self.t_height = sensor_size

        xx, yy = torch.meshgrid(torch.arange(self.t_width), torch.arange(self.t_height))
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        self.xx, self.yy = xx, yy

        self.last_t = None

    def __call__(self, pred_g, trgt_g, trgt, flow, normalize=True):
        mse = torch.mul(self.mse_loss(pred_g, trgt_g), 0.8).to(pred_g.device)
        pred = frankotchellappa(pred_g)

        flow_x = flow[:, 0, :, :]
        flow_y = flow[:, 1, :, :]

        warping_grid_x = self.xx.to(pred.device) - flow_x
        warping_grid_y = self.yy.to(pred.device) - flow_y

        warping_grid_x = (2 * warping_grid_x / (self.t_width - 1)) - 1
        warping_grid_y = (2 * warping_grid_y / (self.t_height - 1)) - 1
        warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)

        image1_warped_to0 = F.grid_sample(trgt, warping_grid, align_corners=True)
        prod1_warped_to0 = F.grid_sample(pred, warping_grid, align_corners=True)

        visibility_mask = torch.exp(-self.alpha * (self.last_t - image1_warped_to0) ** 2)
        tc_loss = visibility_mask * torch.abs(self.last_t - prod1_warped_to0) \
             / (torch.abs(self.last_t) + torch.abs(prod1_warped_to0) + 1e-5)

        lpips_val = self.model(pred, trgt, normalize=normalize).mean().to(pred.device)

        return lpips_val, mse, tc_loss.mean().to(pred.device)


class sparse_loss():
    def __init__(self, args, L0=1, net='alex', alpha=50):
        self.idx_loss = F.mse_loss
        self.model = lpips.LPIPS(net=net).to(args.device)
        self.h = args.crop_size[0]
        self.w = args.crop_size[1]
        self.L0 = L0
        self.alpha = alpha
        xx, yy = torch.meshgrid(torch.arange(self.w), torch.arange(self.h))
        xx = xx.to(args.device)
        yy = yy.to(args.device)
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        self.xx, self.yy = xx.float(), yy.float()

    def __call__(self, i, output, eve_grad, grd_full, flow):
        idx_loss = self.idx_loss(output[..., :4], eve_grad[..., :4])

        g_x = torch.zeros((output.shape[0], self.h * self.w, 1)).type_as(output).to(output.device)
        g_y = torch.zeros((output.shape[0], self.h * self.w, 1)).type_as(output).to(output.device)

        idx = (output[:, :, 1:2] * self.w) + output[:, :, 0:1]

        g_x = g_x.scatter_add_(1, idx.long(), output[:, :, 2:3]).view(output.shape[0], 1, self.h, self.w)
        g_y = g_y.scatter_add_(1, idx.long(), output[:, :, 3:4]).view(output.shape[0], 1, self.h, self.w)

        pred_grad = torch.cat([g_x, g_y, g_x + g_y], dim=1)
        trgt_grad = torch.cat([grd_full, grd_full.sum(1, keepdim=True)], dim=1)

        dist = self.model(pred_grad, trgt_grad)
        loss = (dist.mean() + idx_loss) / 2

        return loss
