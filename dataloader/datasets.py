import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2
import pickle as pkl

from .file_io import read_img, read_disp
from .transforms import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataset(Dataset):
    def __init__(self,
                 transform=None,
                 is_vkitti2=False,
                 is_sintel=False,
                 is_middlebury_eth3d=False,
                 is_tartanair=False,
                 is_instereo2k=False,
                 is_crestereo=False,
                 is_fallingthings=False,
                 is_raw_disp_png=False,
                 half_resolution=False,
                 ):

        super(StereoDataset, self).__init__()

        self.transform = transform
        self.save_filename = False

        self.is_vkitti2 = is_vkitti2
        self.is_sintel = is_sintel
        self.is_middlebury_eth3d = is_middlebury_eth3d
        self.is_tartanair = is_tartanair
        self.is_instereo2k = is_instereo2k
        self.is_crestereo = is_crestereo
        self.is_fallingthings = is_fallingthings
        self.half_resolution = half_resolution
        self.is_raw_disp_png = is_raw_disp_png

        self.samples = []

    def __getitem__(self, index):
        sample = {}
        # file path
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']
        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        if 'disp' in sample_path and sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'],
                                       vkitti2=self.is_vkitti2,
                                       sintel=self.is_sintel,
                                       tartanair=self.is_tartanair,
                                       instereo2k=self.is_instereo2k,
                                       fallingthings=self.is_fallingthings,
                                       crestereo=self.is_crestereo,
                                       raw_disp_png=self.is_raw_disp_png,
                                       )  # [H, W]

            # for middlebury and eth3d datasets, invalid is denoted as inf
            if self.is_middlebury_eth3d or self.is_crestereo:
                sample['disp'][sample['disp'] == np.inf] = 0

        if self.half_resolution:
            sample['left'] = cv2.resize(sample['left'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            sample['right'] = cv2.resize(sample['right'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            sample['disp'] = cv2.resize(sample['disp'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) * 0.5

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples

        return self


class FlyingThings3D(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/flyingthings3d/',
                 mode='TRAIN',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(FlyingThings3D, self).__init__(transform=transform)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            # samples: train: 22390, test: 4370
            left_files = sorted(glob(data_dir + '/' + split + '/' + mode + '/*/*/left/*.png'))

            for left_name in left_files:
                sample = dict()

                sample['left'] = left_name
                sample['right'] = left_name.replace('/left/', '/right/')
                sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

                self.samples.append(sample)
            # with open(data_dir + '/sample.pkl', 'ab') as f:
            #     pkl.dump(self.samples, f)

class FlyingThings3D_oneQuat(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/flyingthings3d_one_quarter/',
                 mode='TRAIN',
                 transform=None,
                 ):
        super(FlyingThings3D_oneQuat, self).__init__(transform=transform)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            # samples: train: 22390, test: 4370
            left_files = sorted(glob(data_dir + '/' + mode + '/*/*/*/*L.png'))

            for left_name in left_files:
                sample = dict()

                sample['left'] = left_name
                sample['right'] = left_name.replace('L', 'R')
                sample['disp'] = left_name.replace('L.png', 'L.pfm').replace('Cleanpass', 'Disparity')

                self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)

class Monkaa(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/monkaa/',
                 split='frames_cleanpass',
                 transform=None,
                 ):
        super(Monkaa, self).__init__(transform=transform)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            # samples: 8664
            left_files = sorted(glob(data_dir + '/' + split + '/*/left/*.png'))

            for left_name in left_files:
                sample = dict()

                sample['left'] = left_name
                sample['right'] = left_name.replace('/left/', '/right/')
                sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

                self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)


class Driving(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/driving/',
                 split='frames_cleanpass',
                 transform=None,
                 ):
        super(Driving, self).__init__(transform=transform)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            # samples: 4400
            left_files = sorted(glob(data_dir + '/' + split + '/*/*/*/left/*.png'))

            for left_name in left_files:
                sample = dict()

                sample['left'] = left_name
                sample['right'] = left_name.replace('/left/', '/right/')
                sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

                self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)

class SintelStereo(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/sintel_stereo_training/',
                 split='clean',
                 transform=None,
                 save_filename=False,
                 ):
        super(SintelStereo, self).__init__(transform=transform, is_sintel=True)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            self.save_filename = save_filename
            assert split in ['clean', 'final']

            # total: clean & final each 1064
            left_files = sorted(glob(data_dir +  split + '_left/*/*.png'))
            right_files = sorted(glob(data_dir +  split + '_right/*/*.png'))
            disp_files = sorted(glob(data_dir + '/disparities/*/*.png'))

            assert len(left_files) == len(right_files) == len(disp_files)
            num_samples = len(left_files)

            for i in range(num_samples):
                sample = dict()

                sample['left'] = left_files[i]
                sample['right'] = right_files[i]
                sample['disp'] = disp_files[i]

                if self.save_filename:
                    sample['left_name'] = left_files[i]

                self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)

class Middlebury20052006(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/middlebury/middlebury20052006/',
                 transform=None,
                 save_filename=False,
                 ):
        super(Middlebury20052006, self).__init__(transform=transform, is_raw_disp_png=True)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            self.save_filename = save_filename

            dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

            for curr_dir in dirs:
                # Middlebury/2005/Art
                sample = dict()

                sample['left'] = os.path.join(data_dir, curr_dir, 'view1.png')
                sample['right'] = os.path.join(data_dir, curr_dir, 'view5.png')
                sample['disp'] = os.path.join(data_dir, curr_dir, 'disp1.png')

                if save_filename:
                    sample['left_name'] = sample['left']

                self.samples.append(sample)

                # same disp for different images
                # gt_disp = os.path.join(data_dir, curr_dir, 'disp1.png')

                # also include different illuminations
                # for illum in ['Illum1', 'Illum2', 'Illum3']:
                #     for exp in ['Exp0', 'Exp1', 'Exp2']:
                #         # Middlebury/2005/Art/Illum1/Exp0/
                #         sample = dict()

                #         sample['left'] = os.path.join(data_dir, curr_dir, illum, exp, 'view1.png')
                #         sample['right'] = os.path.join(data_dir, curr_dir, illum, exp, 'view5.png')
                #         sample['disp'] = gt_disp

                #         if save_filename:
                #             sample['left_name'] = sample['left']

                #         self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)

class Middlebury2014(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/middlebury/middlbury2014/',
                 transform=None,
                 save_filename=False,
                 half_resolution=True,
                 ):
        super(Middlebury2014, self).__init__(transform=transform, is_middlebury_eth3d=True,
                                             half_resolution=half_resolution,
                                             )
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            self.save_filename = save_filename
            all_contents = os.listdir(data_dir)

            dirs = [folder for folder in all_contents if os.path.isdir(os.path.join(data_dir, folder))]
            # dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]
            print(dirs)
            for curr_dir in dirs:
                # for data_type in ['', 'E', 'L']:
                for data_type in ['']:
                    sample = dict()

                    sample['left'] = os.path.join(data_dir, curr_dir, 'im0.png')
                    sample['right'] = os.path.join(data_dir, curr_dir, 'im1' + '%s.png' % data_type)
                    sample['disp'] = os.path.join(data_dir, curr_dir, 'disp0.pfm')

                    if save_filename:
                        sample['left_name'] = sample['left']

                    self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)

class CREStereoDataset(StereoDataset):
    def __init__(self,
                 data_dir='/home/zhengzihua/nas_public/zhengzihua/dataset/crestereo/',
                 transform=None,
                 ):
        super(CREStereoDataset, self).__init__(transform=transform, is_crestereo=True)
        pkl_name = data_dir + 'sample.pkl'
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self.samples = pkl.load(f)
        else:
            assert(os.path.exists(data_dir))
            left_files = sorted(glob(data_dir + '/*/*_left.jpg'))
            right_files = sorted(glob(data_dir + '/*/*_right.jpg'))
            disp_files = sorted(glob(data_dir + '/*/*_left.disp.png'))

            assert len(left_files) == len(right_files) == len(disp_files)
            num_samples = len(left_files)

            for i in range(num_samples):
                sample = dict()

                sample['left'] = left_files[i]
                sample['right'] = right_files[i]
                sample['disp'] = disp_files[i]

                self.samples.append(sample)
            with open(data_dir + '/sample.pkl', 'ab') as f:
                pkl.dump(self.samples, f)


def build_dataset(args):
    if args.stage == 'pretrain':
        train_transform_list = [RandomScale(min_scale=0,
                                                       max_scale=1.0,
                                                       crop_width=args.image_width),
                                RandomCrop(args.image_height, args.image_width),
                                RandomRotateShiftRight(),
                                RandomColor(),
                                RandomVerticalFlip(),
                                ToTensor(),
                                # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = Compose(train_transform_list)
        
        things = FlyingThings3D(transform=train_transform) # 22390
        things_oq = FlyingThings3D_oneQuat(transform=train_transform) #23570

        monkaa = Monkaa(transform=train_transform) #8664
        driving = Driving(transform=train_transform) #4400

        sintel = SintelStereo(transform=train_transform) #1064

        # high res data transform
        train_transform_list = [RandomScale(min_scale=-0.2,
                                                       max_scale=0.4,
                                                       crop_width=args.image_width,
                                                       nearest_interp=True,
                                                       ),
                                RandomCrop(args.image_height, args.image_width),
                                RandomRotateShiftRight(),
                                RandomColor(),
                                RandomVerticalFlip(),
                                ToTensor(),
                                # Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = Compose(train_transform_list)

        crestereo = CREStereoDataset(transform=train_transform) # 200000


        mb20052006 = Middlebury20052006(transform=train_transform) # 270
        # return mb20052006

        train_dataset =  5 * things + things_oq + monkaa + driving + 5 * sintel + \
                        crestereo + 200 * mb20052006

        return train_dataset

    

    else:
        raise NotImplementedError

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--stage', default='pretrain', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--image_height', default=512, type=int, help='')
    parser.add_argument('--image_width', default=768, type=int, help='')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    train_data = build_dataset(args)
    # for i in range(10):
    sample = train_data[100000]
    # read images
    left_image = sample['left']
    right_image = sample['right']
    disparity_map = sample['disp']
    left_image = torch.squeeze(left_image.permute(1, 2, 0)).cpu().detach().numpy()
    right_image = torch.squeeze(right_image.permute(1, 2, 0)).cpu().detach().numpy()
    disparity_map = torch.squeeze(disparity_map).cpu().detach().numpy()
    print(disparity_map.max(), disparity_map.min())
    height, width = disparity_map.shape

    y, x = np.indices((height, width), dtype=np.float32)

    new_x = x - disparity_map

    new_x = np.clip(new_x, 0, width - 1)

    remapped_right_image = cv2.remap(right_image, new_x.astype(np.float32), y, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    print('****len:', len(train_data))
    cv2.imwrite("left.jpg", left_image)
    cv2.imwrite("remapped_right_image.jpg", remapped_right_image)
    disp = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    cv2.imwrite("disp.jpg", disp)