from genericpath import exists
import glob
import argparse
import math
import random
import os
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/rsna_data',
                    help='trainig image saving directory')
parser.add_argument('--ratio', type=float, default=0.2, help='validation data ratio')
args = parser.parse_args()

if __name__ == '__main__':
    src_img_dir = os.path.join(args.data_root, 'images/all_train')
    data_size = len(glob.glob1(src_img_dir, "*.png"))
    valid_size = math.floor(data_size * args.ratio)

    img_list = []
    for img_path in glob.glob(f'{src_img_dir}/*.png'):
        img_list.append(img_path)

    idx = random.sample(range(data_size), valid_size)

    dest_img_dir = os.path.join(args.data_root, 'images')
    train_img_dir = os.path.join(dest_img_dir, 'train')
    valid_img_dir = os.path.join(dest_img_dir, 'val')
    src_label_dir = os.path.join(args.data_root, 'annotations/all_train')
    train_label_dir = src_label_dir.replace('all_train', 'train')
    valid_label_dir = src_label_dir.replace('all_train', 'val')
    # if not os.path.isdir(dest_img_dir):
    os.makedirs(train_img_dir,exist_ok=True)
    os.makedirs(valid_img_dir,exist_ok=True)
    os.makedirs(train_label_dir,exist_ok=True)
    os.makedirs(valid_label_dir,exist_ok=True)

    pbar=tqdm(range(data_size))
    for i in pbar:
        pbar.set_description(img_list[i])

        if i in idx:
            src_img = img_list[i]
            dest_img = src_img.replace('all_train', 'val')
            shutil.copy(src_img, dest_img)
            src_label = src_img.replace('images', 'annotations').replace('png', 'txt')
            dest_label = src_label.replace('all_train', 'val')
            shutil.copyfile(src_label, dest_label)
        else:
            src_img = img_list[i]
            dest_img = src_img.replace('all_train', 'train')
            shutil.copy(src_img, dest_img)
            src_label = src_img.replace('images', 'annotations').replace('png', 'txt')
            dest_label = src_label.replace('all_train', 'train')
            shutil.copyfile(src_label, dest_label)

    train_size = len(glob.glob1(train_img_dir, "*.png"))
    valid_size = len(glob.glob1(valid_img_dir, "*.png"))
    print(f'train size: {train_size}\tvalid size: {valid_size}')