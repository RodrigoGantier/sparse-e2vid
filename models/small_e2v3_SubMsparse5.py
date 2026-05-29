import spconv.pytorch as spconv
from spconv.pytorch.core import SparseConvTensor
from spconv.core import ConvAlgo
import torch
import torch.nn as nn
from functools import reduce
from typing import List
from einops import rearrange
from tools.timers import CudaTimer_ma


def std_sample_sparse(ten):

    features = ten.features.sum(1)
    ten_mean = features.mean()
    ten_std = features.std() 
    v_up = ten_mean + ten_std * 0.3
    v_down = ten_mean - ten_std * 0.3

    sample_idx = 1 -((v_down<features) & (features<v_up)).float()
    sample_idx = sample_idx.bool()

    ten_vals = ten.features[sample_idx, :]
    ten_idx = ten.indices[sample_idx, :]

    res = SparseConvTensor(ten_vals,
                           ten_idx,
                           ten.spatial_shape,
                           ten.batch_size,
                           benchmark=ten.benchmark)
    return res


def rand_sample_sparse(ten):

    num_sample = ten.features.shape[0]
    sample_idx = torch.randperm(num_sample)[:int(num_sample * 0.00)]

    res_shape = [
        ten.batch_size, 
        *ten.spatial_shape, 
        ten.features.shape[1]
    ]

    ten_ths = torch.sparse_coo_tensor(ten.indices.T,
                                    ten.features,
                                    res_shape,
                                    requires_grad=False).coalesce()

    c_th_inds = ten_ths.indices()[:, sample_idx]
    c_th_inds = c_th_inds.T.contiguous().int()
    c_th_values = ten_ths.values()[sample_idx, :]

    res = SparseConvTensor(c_th_values,
                           c_th_inds,
                           ten.spatial_shape,
                           ten.batch_size,
                           benchmark=ten.benchmark)
    return res


def sparse_cat(t1, t2):
    ch_dim =  t1.features.shape[1] + t2.features.shape[1]
    fx = torch.nn.functional.pad(t1.features, (0, t2.features.shape[1]), "constant", 0)
    fp = torch.nn.functional.pad(t2.features, (t1.features.shape[1], 0), "constant", 0)


    features = torch.cat([fx, fp], 0)
    indices = torch.cat([t1.indices, t2.indices], 0).contiguous().int()

    c_th = torch.sparse_coo_tensor(
        indices.T, features, 
        [t1.batch_size, *t1.spatial_shape, ch_dim], 
        requires_grad=True).coalesce()

    c_th_inds = c_th.indices().T.contiguous().int()
    c_th_values = c_th.values()
    assert c_th_values.is_contiguous()

    res = SparseConvTensor(c_th_values,
                           c_th_inds,
                           t1.spatial_shape,
                           t1.batch_size,
                           benchmark=t1.benchmark)
    
    res.benchmark_record = t1.benchmark_record
    res._timer = t1._timer
    res.thrust_allocator = t1.thrust_allocator
    return res



def sp_multiply(*tensors: SparseConvTensor):

    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    res_shape = [
        tensors[0].batch_size, 
        *tensors[0].spatial_shape, 
        tensors[0].features.shape[1]
    ]

    for i, ten in enumerate(tensors):
        assert ten.spatial_shape == tensors[0].spatial_shape
        assert ten.batch_size == tensors[0].batch_size
        assert ten.features.shape[1] == tensors[0].features.shape[1]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(
            torch.sparse_coo_tensor(ten.indices.T,
                                    ten.features,
                                    res_shape,
                                    requires_grad=True))

    c_th = reduce(lambda x, y: torch.multiply(x, y), ten_ths).coalesce()

    c_th_inds = c_th.indices().reshape(tensors[0].indices.shape[1], c_th.indices().shape[1]).T.contiguous().int()
    c_th_values = c_th.values().reshape(c_th.values().shape[0], tensors[0].features.shape[1])
    assert c_th_values.is_contiguous()

    res = SparseConvTensor(c_th_values,
                           c_th_inds,
                           tensors[0].spatial_shape,
                           tensors[0].batch_size,
                           benchmark=tensors[0].benchmark)
    
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tensors[max_num_indices_idx].indice_dict

    res.benchmark_record = tensors[0].benchmark_record
    res._timer = tensors[0]._timer
    res.thrust_allocator = tensors[0].thrust_allocator
    return res


def sp_add(*tens: SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique 
    """
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    first = tens[0]
    res_shape = [
        first.batch_size, *first.spatial_shape, first.features.shape[1]
    ]

    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] == tens[0].features.shape[1]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(
            torch.sparse_coo_tensor(ten.indices.T,
                                    ten.features,
                                    res_shape,
                                    requires_grad=True))

    c_th = reduce(lambda x, y: x + y, ten_ths).coalesce()

    c_th_inds = c_th.indices().reshape(c_th.indices().shape).T.contiguous().int()
    c_th_values = c_th.values().reshape(c_th.values().shape)
    assert c_th_values.is_contiguous()

    res = SparseConvTensor(c_th_values,
                           c_th_inds,
                           first.spatial_shape,
                           first.batch_size,
                           benchmark=first.benchmark)
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer
    res.thrust_allocator = first.thrust_allocator
    return res


def spOneSub(ten):
    features = 1 - ten.features
    ten = ten.replace_feature(features)
    return ten


class ConvLayer(nn.Module):
    def __init__(self, in_chans, out_chans, k, s):
        super().__init__()

        # p = k // 2
        p = 0
        self.depth_point_wise = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=k, padding=p, stride=s, groups=in_chans),
            nn.Conv2d(in_chans, out_chans, kernel_size=1))

    def forward(self, x):
        return self.depth_point_wise(x)


class convGru(nn.Module):
    """
    Convolutional GRU cell
    """
    def __init__(self, input_size, hidden_size, k=1):
        super(convGru, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        p = k//2

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=p, )
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=p, )
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, k, padding=p, )

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)

        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

        # self.reset_gate = ConvLayer(input_size + hidden_size, hidden_size, k, s=1)
        # self.update_gate = ConvLayer(input_size + hidden_size, hidden_size, k, s=1)
        # self.out_gate = ConvLayer(input_size + hidden_size, hidden_size, k, s=1)

        # nn.init.orthogonal_(self.reset_gate.depth_point_wise[0].weight)
        # nn.init.orthogonal_(self.reset_gate.depth_point_wise[1].weight)

        # nn.init.orthogonal_(self.update_gate.depth_point_wise[0].weight)
        # nn.init.orthogonal_(self.update_gate.depth_point_wise[1].weight)

        # nn.init.orthogonal_(self.out_gate.depth_point_wise[0].weight)
        # nn.init.orthogonal_(self.out_gate.depth_point_wise[1].weight)


        # nn.init.constant_(self.reset_gate.depth_point_wise[0].bias, 0.0)
        # nn.init.constant_(self.reset_gate.depth_point_wise[1].bias, 0.0)

        # nn.init.constant_(self.update_gate.depth_point_wise[0].bias, 0.0)
        # nn.init.constant_(self.update_gate.depth_point_wise[1].bias, 0.0)

        # nn.init.constant_(self.out_gate.depth_point_wise[0].bias, 0.0)
        # nn.init.constant_(self.out_gate.depth_point_wise[1].bias, 0.0)
        

        self.prev_state = None
        
    def forward(self, x):

        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        # generate empty prev_state, if None is provided
        if self.prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            self.prev_state = torch.zeros(state_size, dtype=x.dtype).to(x.device)
        # else:
        #     mask = self.prev_state[0].sum(0)
        #     mask -= mask.median()
        #     non_zeros = torch.sum(mask != 0)
        #     mask_mean = mask.sum() / non_zeros
        #     mask_std = torch.sqrt((mask ** 2).sum() / non_zeros - mask_mean ** 2)
        #     v_up = mask_mean + mask_std * 0.1
        #     v_down = mask_mean - mask_std * 0.1
        #     m = torch.ones_like(mask)
        #     m[(v_down<mask) & (mask<v_up)] = 0
        #     self.prev_state *= m 

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([x, self.prev_state], dim=1)
        # stacked_inputs = torch.nn.functional.pad(stacked_inputs, (1, 1, 1, 1), mode='reflect')
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, self.prev_state * reset], dim=1)))
        new_state = self.prev_state * (1 - update) + out_inputs * update
        self.prev_state = new_state

        return new_state


class sparseBlock(nn.Module):
    """
    sparseBlock 
    Default: bias, GeLU, no downsampling.
    """

    def __init__(self, in_channels, out_channels, k):
        super(sparseBlock, self).__init__()
        
        p = k//2
        self.conv = spconv.SparseSequential(

            spconv.SubMConv2d(in_channels, out_channels*2, k, padding=p, algo=None),
            nn.GELU(),

            spconv.SubMConv2d(out_channels*2, out_channels*2, k, padding=p, algo=None),
            nn.GELU(),
        
            spconv.SubMConv2d(out_channels*2, out_channels*2, k, padding=p, algo=None),
            nn.GELU(),

            spconv.SubMConv2d(out_channels*2, out_channels, k, padding=p, algo=None),
            nn.GELU(), spconv.ToDense())
        
    def forward(self, x):
        
        x = self.conv(x) 
        return x


class outputLayer(nn.Module):
    def __init__(self, in_chans, out_chans, k):
        super().__init__()
        
        self.conv0 = nn.Conv2d(in_chans, out_chans, kernel_size=k, padding_mode='reflect')
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv0(x)
        x = self.act(x)
        return x


class e2v(nn.Module):
    def __init__(self, args):
        super().__init__()

        k = args.kernel_size
        
        self.sparse = sparseBlock(args.in_chans, args.embed_dim, k=k)
        self.G0 = convGru(args.embed_dim, args.embed_dim, k=k)
        self.output = outputLayer(args.embed_dim, args.out_chans, k=1)
       
        self.sparse_t = CudaTimer_ma('sparse_t')
        self.G0_t = CudaTimer_ma('G0_t')
        self.output_t = CudaTimer_ma('output_t')


    def reset_states(self, ):
        self.G0.prev_state = None

    def forward(self, input):

        # sparse
        # with self.sparse_t:
        #     input = rearrange(input, 'b c h w -> b h w c')
        #     input = spconv.SparseConvTensor.from_dense(input)
        #     x = self.sparse(input)
        input = rearrange(input, 'b c h w -> b h w c')
        input = spconv.SparseConvTensor.from_dense(input)
        x = self.sparse(input)

        # dense recurrent 
        #  with self.G0_t:
        #      x = self.G0(x)
        x = self.G0(x)

        # dense output
        # with self.output_t:
        #     x = self.output(x)
        x = self.output(x)

        return x