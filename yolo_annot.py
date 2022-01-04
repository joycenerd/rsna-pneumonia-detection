import argparse
import csv
import os
from pydicom import dcmread
from tqdm import tqdm
import shutil
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, help='data root path')
parser.add_argument('--destroot', type=str, help='data root path')
args = parser.parse_args()


if __name__ == '__main__':
    label_path = os.path.join(args.dataroot, 'stage_2_train_labels.csv')
    label_f = open(label_path)
    csvreader = csv.reader(label_f)
    header = next(csvreader)
    train_dir = os.path.join(args.dataroot, 'stage_2_train_images')
    annot_dir = os.path.join(args.destroot, 'labels/all_train')
    dest_img_dir = os.path.join(args.destroot, 'images/all_train')
    if not os.path.isdir(annot_dir):
        os.makedirs(annot_dir)
    if not os.path.isdir(dest_img_dir):
        os.makedirs(dest_img_dir)
    cnt = 0
    prev_patient_id = '0'
    annot_path = None
    pbar = tqdm(csvreader)
    for row in pbar:
        target = int(row[5])
        if target == 0:
            cnt += 1
            continue
        patient_id, x, y = row[0], float(row[1]), float(row[2])
        width, height = float(row[3]), float(row[4])
        pbar.set_description(patient_id)

        # get image information
        img_path = os.path.join(train_dir, f'{patient_id}.dcm')
        ds = dcmread(img_path)
        w, h = ds.Columns, ds.Rows
        # convert to yolo annotation
        obj_class = 0
        x_center = (x+width/2) / w
        y_center = (y+height/2) / h
        width = width / w
        height = height / w
        label_str = f"{obj_class} {x_center} {y_center} {width} {height}\n"
        # write to file
        if patient_id == prev_patient_id:
            f.write(label_str)
        else:
            if annot_path is not None:
                f.close()
            annot_path = os.path.join(annot_dir, f"{patient_id}.txt")
            f = open(annot_path, mode='w')
            f.write(label_str)
            prev_patient_id = patient_id
            # move image (separate useful image from others) and convert to png
            img = ds.pixel_array
            cv2.imwrite(os.path.join(dest_img_dir, f'{patient_id}.png'), img)
    print(f"no detected object count: {cnt}")
