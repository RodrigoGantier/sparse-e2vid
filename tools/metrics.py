import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_small
from torchvision.models import VGG19_Weights
from collections import deque
from pytorch_msssim import SSIM


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class metrics_Fn:
    def __init__(self, sensor_size, to_cuda):
        self.device = to_cuda
        self.vgg = VGG19().to(to_cuda).eval()
        self.MSE = nn.MSELoss()
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.flownet = raft_small(weights=weights, progress=False).to(to_cuda).eval()
        self.alpha = 50

        self.t_height, self.t_width = sensor_size
        xx, yy = torch.meshgrid(torch.arange(self.t_width), torch.arange(self.t_height), indexing='ij')
        xx.transpose_(0, 1)
        yy.transpose_(0, 1)
        self.xx, self.yy = xx.to(to_cuda), yy.to(to_cuda)
        self.pred0 = None
        self.img0 = None

    def temp_loss(self, pred1, img1):
        with torch.no_grad():
            img0_, img1_ = self.transforms(self.img0/255.0, img1/255.0)
            img0_ = F.pad(img0_, (0, 0, 2, 2)).to(self.device).float()
            img1_ = F.pad(img1_, (0, 0, 2, 2)).to(self.device).float()
            flow = self.flownet(img0_, img1_)[-1]
            flow = flow[:, :, 2:-2, :]
            flow_x = flow[:, 0, :, :].to(self.device)
            flow_y = flow[:, 1, :, :].to(self.device)

            warping_grid_x = self.xx - flow_x
            warping_grid_y = self.yy - flow_y

            warping_grid_x = (2 * warping_grid_x / (self.t_width - 1)) - 1
            warping_grid_y = (2 * warping_grid_y / (self.t_height - 1)) - 1
            warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)

            image1_warped_to0 = F.grid_sample(img1, warping_grid, align_corners=True)
            prod1_warped_to0 = F.grid_sample(pred1, warping_grid, align_corners=True)

            visibility_mask = torch.exp(-self.alpha * (self.img0 - image1_warped_to0) ** 2)
            tc_loss = visibility_mask * torch.abs(self.pred0 - prod1_warped_to0) / (torch.abs(self.pred0) + torch.abs(prod1_warped_to0) + 1e-5)

            self.pred0 = pred1.clone()
            self.img0 = img1.clone()
        return tc_loss.mean().item()

    def lpips_fn(self, pred, y):
        y = ((y/255.0) * 2) - 1
        pred = ((pred/255.0) * 2) - 1
        x_vgg, y_vgg = self.vgg(pred), self.vgg(y.detach())
        f_loss = 0
        for i in range(len(x_vgg)):
            f_loss += self.weights[i] * self.MSE(x_vgg[i], y_vgg[i].detach())
        return f_loss.item() / len(x_vgg)

    def ssim_metric(self, pred, y):
        return 1 - self.ssim_module(pred/255.0, y/255.0).item()

    def mse_metric(self, pred, y):
        return self.MSE(pred/255.0, y/255.0).item()
