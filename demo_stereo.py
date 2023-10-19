import os
import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from nets_stereo.dip import DIP
from nets_stereo.utils.utils import InputPadder
from imread_from_url import imread_from_url

out_path = 'demo-outputs/'
DEVICE = 'cuda'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
    print('image_files:', image_files)
    return images
#Ref: https://github.com/megvii-research/CREStereo/
def demo_CRE_test(args):
    imgL = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
    imgR = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

    # Resize image in case the GPU memory overflows
    # in_h = 768
    # in_w = 1024
    # eval_h, eval_w = (in_h,in_w)
    # assert eval_h%16 == 0, "input height should be divisible by 8"
    # assert eval_w%16 == 0, "input width should be divisible by 8"

    # imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    imgL = imgL.transpose(2, 0, 1)
    imgR = imgR.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])
    imgL = torch.tensor(imgL.astype("float32")).to(DEVICE)
    imgR = torch.tensor(imgR.astype("float32")).to(DEVICE)

    #define model
    model = DIP(max_disp=args.max_disp, mixed_precision=False, test_mode=True)
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    pre_train = torch.load(args.model)
    model.load_state_dict(pre_train, strict=False)
    model.eval()

    padder = InputPadder(imgL.shape)
    imgL, imgR = padder.pad(imgL, imgR)
    with torch.no_grad():
        pred_flow = model(imgL, imgR, iters=4)
    disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()
    # t = float(in_w) / float(eval_w)
    # disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(out_path + "output.jpg", disp_vis)

def demo(args):
    model = DIP(max_disp=args.max_disp, mixed_precision=False, test_mode=True)
    model = torch.nn.DataParallel(model)
    model.cuda()
    pre_train = torch.load(args.model)
    model.load_state_dict(pre_train, strict=False)

    model.eval()
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = load_image_list(images)
        print('len images:', len(images))
        for i in range(len(images)//2):
            image1 = images[i*2]
            image2 = images[i*2+1]
            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)
            print('image1.shape: ', image1.shape, image2.shape)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_up = model(image1, image2, iters=4)

            flow_up  = padder.unpad(flow_up)
            disp = flow_up[0, 0, :, :].cpu().numpy()
            
            disp[disp < 0] = 0

            norm_disp = cv2.normalize(disp, None, alpha=255, beta=0.0, norm_type=cv2.NORM_MINMAX)
            cv2.imwrite(out_path + str(i+1) + '_disp.jpg', norm_disp)
            print("i: ",i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--test_cre', action='store_true')
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--max_disp', type=float, default=256)
    args = parser.parse_args()

    out_path = 'demo-outputs/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.test_cre:
        demo_CRE_test(args)
    else:
        demo(args)