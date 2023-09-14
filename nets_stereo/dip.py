import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, BasicEncoderQuarter
from .utils.utils import bilinear_sampler, coords_grid, upflow4
from .path_match import PathMatch

import argparse

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class DIP(nn.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super(DIP, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128

        self.dropout = 0
        self.iters = 5

        # feature network, context network, and update block
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=self.dropout)
        # self.cnet = BasicEncoderQuarter(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)

        self.update_block_s = SmallUpdateBlock(hidden_dim=self.hidden_dim)
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def random_init_flow(self, fmap, max_flow):
        N, C, H, W = fmap.shape

        flow_u = - torch.rand(N, 1, H, W) * max_flow
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)

        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def upsample_flow(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def forward(self, image1, image2, iters = 4):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # 1/4
        patch_fn = PathMatch(fmap1, fmap2)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            # cnet = self.cnet(image1)
            net, inp = torch.split(fmap1, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # 1/4 -> 1/16
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            s_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)
            # context(left)
            s_net = F.avg_pool2d(net, 4, stride=4)
            s_inp = F.avg_pool2d(inp, 4, stride=4)

        # 1/16
        s_patch_fn = PathMatch(s_fmap1, s_fmap2)

        # init flow
        s_flow = self.random_init_flow(s_fmap1, max_flow=self.max_flow // 16)

        # small refine: 1/16
        flow = None
        flow_up = None
        flow_predictions = []
        for itr in range(iters):
            # --------------- update1 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_shift=False, shift=2)

            with autocast(enabled=self.mixed_precision):
                s_net, up_mask, delta_flow = self.update_block_s(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up = - upflow4(flow)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_shift=True, shift=2)

            with autocast(enabled=self.mixed_precision):
                s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up = - upflow4(flow)
            flow_predictions.append(flow_up)

        # large refine: 1/4
        for itr in range(iters):
            # --------------- update1 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_shift=False, shift=2)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block_s(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = - self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_shift=True, shift=2)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = - self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return flow_predictions
