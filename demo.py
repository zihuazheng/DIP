
import re
import io
import os
import argparse
import cv2
import glob
import numpy as np
import torch
from nets.dip import DIP

from nets.utils.utils import InputPadder, forward_interpolate

DEVICE = 'cuda'

def flow_to_image_ndmax(flow, max_flow=256, isBGR2RGB = True):
    # flow shape (H, W, C)
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    im_rgb = (im * 255).astype(np.uint8)
    if isBGR2RGB:
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
    return im_rgb


def warp_cv2(img_prev, flow):
    # calculate mat
    w = int(img_prev.shape[1])
    h = int(img_prev.shape[0])
    flow=np.float32(flow)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    print('pixel_map', pixel_map.shape)
    new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
    return new_frame

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def test_data(args):
    model = DIP(max_offset=256, mixed_precision=False, test_mode=True)
    model = torch.nn.DataParallel(model)
    model.cuda()
    warm_start  = True

    pre_train = torch.load('DIP_sintel.pth')
    model.load_state_dict(pre_train, strict=False)

    model.eval()

    flow_prev = None
    with torch.no_grad():
        for i in range(1, 4):
            img_path1 = './input_imgs/frame_' + str(i).zfill(4) + '.png'
            img_path2 = './input_imgs/frame_' + str(i+1).zfill(4) + '.png'
            
            img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
            img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

            image1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            image1 = image1[None].to(DEVICE)
            image2 = image2[None].to(DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            if flow_prev is None:
                flow_up = model(image1, image2, iters=20, init_flow=None)
            else:
                flow_up = model(image1, image2, iters=12, init_flow=flow_prev)
            if warm_start:
                flow_prev = forward_interpolate(flow_up[0])[None].cuda()
            flow_up  = padder.unpad(flow_up)
            flo = flow_up[0].view(2, flow_up[0].shape[-2], flow_up[0].shape[-1])
            flo = flo.permute(1,2,0).cpu().numpy()

            color_flow = flow_to_image(flo, clip_flow=None, convert_to_bgr=True)

            cv2.imwrite(out_path + str(i) + '_flow.jpg', color_flow)
            print(out_path + "i: ",i)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="DIP_sintel.pth")
    parser.add_argument('--max_flow', type=float, default=256)

    args = parser.parse_args()

    out_path = 'demo-outputs/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    test_data(args)