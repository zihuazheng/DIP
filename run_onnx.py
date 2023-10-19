import os
import argparse
import cv2
import glob
import numpy as np
from PIL import Image
from dip_onnx_runner import DIPRunner

out_path = 'demo-outputs/'
DEVICE = 'cuda'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img[None]

def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
    print('image_files:', image_files)
    return images

def demo(args):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    runner = DIPRunner(
        DIP_path=args.DIP_path,
        providers=providers,
    )
    
    images = glob.glob(os.path.join(args.path, '*.png')) + \
                glob.glob(os.path.join(args.path, '*.jpg'))

    images = load_image_list(images)
    print('len images:', len(images))
    for i in range(len(images)//2):
        image1 = images[i*2]
        image2 = images[i*2+1]
        # image1 = image1.to(DEVICE)
        # image2 = image2.to(DEVICE)
        print('image1.shape: ', image1.shape, image2.shape)
        # padder = InputPadder(image1.shape)
        # image1, image2 = padder.pad(image1, image2)
        res = runner.run(image1, image2)[0].astype(np.int64)
        disp = res[0, 0, :, :]
        
        disp[disp < 0] = 0

        norm_disp = cv2.normalize(disp, None, alpha=255, beta=0.0, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(out_path + str(i+1) + '_disp_onnx.jpg', norm_disp)
        print("i: ",i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DIP_path', help="DIP ONNX path")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--max_disp', type=float, default=256)
    args = parser.parse_args()

    out_path = 'demo-outputs/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    demo(args)