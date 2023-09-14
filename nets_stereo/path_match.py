import torch
import torch.nn.functional as F
from .utils.utils import coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class PathMatch:
    def __init__(self, fmap1, fmap2):
        self.map1 = fmap1
        self.map2 = fmap2
        self.N, self.C, self.H, self.W = fmap1.shape
        self.single_planes = self.C // 2
        self.splitShape = [self.single_planes, self.single_planes]
        self.coords = coords_grid(self.N, self.H, self.W).to(fmap1.device)
        self.shift_map2 = self.gen_shift_map2(fmap2)
        self.view_map1 = fmap1.view(self.N, self.C//2, 2, 1, self.H, self.W)
    def gen_shift_map2(self, fmap2):
        # 
        fmap2_tl = F.pad(fmap2, [2, 0, 2, 0], mode = 'replicate')[:, :, 0:self.H, 0:self.W]
        fmap2_tr = F.pad(fmap2, [0, 2, 2, 0], mode = 'replicate')[:, :, 0:self.H, 2:]
        fmap2_dl = F.pad(fmap2, [2, 0, 0, 2], mode = 'replicate')[:, :, 2:, 0:self.W]
        fmap2_dr = F.pad(fmap2, [0, 2, 0, 2], mode = 'replicate')[:, :, 2:, 2:]
        return torch.cat((fmap2, fmap2_tl, fmap2_tr, fmap2_dl, fmap2_dr), dim=1)

    def warp(self, coords, image, h, w):
        # scale grid to [-1,1]
        coords[: ,0 ,: ,:] = 2.0 *coords[: ,0 ,: ,:].clone() / max(self.W -1 ,1 ) -1.0
        coords[: ,1 ,: ,:] = 2.0 *coords[: ,1 ,: ,:].clone() / max(self.H -1 ,1 ) -1.0

        coords = coords.permute(0 ,2 ,3 ,1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
        return output

    def search(self, flow, scale = 1):
        corrs = []
        # u, v = torch.split(flow, [1, 1], dim=1)

        temp_coord = self.coords + flow
        map2_warp = self.warp(temp_coord, self.map2, self.H, self.W)
        padd_map2 = F.pad(map2_warp, [2, 2, 0, 0], mode = 'replicate')
        # #current
        # random search
        for i in range(5):
            map2 = padd_map2[:, :, :, i:i+self.W]
            corr = torch.split(self.map1 * map2, self.splitShape, dim=1)
            for j in range(len(corr)):
                cost = torch.mean(corr[j], dim=1,keepdim=True)
                corrs.append(cost)

        out_corrs = torch.cat(corrs, dim=1)
        return out_corrs

    def inverse_propagation(self, flow):
        corrs = []
        temp_coord = self.coords + flow
        map2_warp = self.warp(temp_coord, self.shift_map2, self.H, self.W)

        map2_warp = map2_warp.view(self.N, self.C//2, 2, 5, self.H, self.W)
        corr = torch.mean(map2_warp * self.view_map1, dim=1)
        corr  = corr.view(self.N, 10, self.H, self.W)
        return corr

    def __call__(self, flow, is_shift = True, shift = 2, scale = 1):
        # print('self.coords.shape: ', self.coords.shape)
        # print('flow.shape: ', flow.shape)
        if(is_shift):
            out_corrs = self.inverse_propagation(flow)
        else:
            out_corrs = self.search(flow, scale = scale)

        return out_corrs