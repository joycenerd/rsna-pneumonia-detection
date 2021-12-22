import argparse
import os
import glob
import cv2
from pydicom import dcmread
from tqdm import tqdm


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/rsna_data',help='data root path')
args=parser.parse_args()


if __name__=='__main__':
    src_imgroot=os.path.join(args.dataroot,'stage_2_test_images')
    dest_imgroot=os.path.join(args.dataroot,'images/test')
    if not os.path.isdir(dest_imgroot):
        os.makedirs(dest_imgroot)
    
    pbar=tqdm(glob.glob(f"{src_imgroot}/*.dcm"))
    for src_img_path in pbar:
        ds=dcmread(src_img_path)
        img=ds.pixel_array

        dest_img_path=src_img_path.replace('stage_2_test_images','images/test').replace('dcm','png')
        cv2.imwrite(dest_img_path,img)