import argparse
import os
import glob
import cv2
from pydicom import dcmread
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, help='data root path')
parser.add_argument('--mode', type=str, default='train', help='train or test')
args = parser.parse_args()


if __name__ == '__main__':
    mode = args.mode
    src_imgroot = os.path.join(args.dataroot, f'stage_2_{mode}_images')
    dest_imgroot = os.path.join(args.dataroot, f'images/{mode}')
    if not os.path.isdir(dest_imgroot):
        os.makedirs(dest_imgroot)
    pbar = tqdm(glob.glob(f"{src_imgroot}/*.dcm"))
    for src_img_path in pbar:
        ds = dcmread(src_img_path)
        img = ds.pixel_array

        dest_img_path = src_img_path.replace(
            f'stage_2_{mode}_images', f'images/{mode}').replace('dcm', 'png')
        cv2.imwrite(dest_img_path, img)
