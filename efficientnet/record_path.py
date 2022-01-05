import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str,
                    default='/eva_data/zchin/rsna_data_all', help='raw data dir')
args = parser.parse_args()


if __name__ == '__main__':
    # train data
    annot_dir = os.path.join(args.dataroot, 'annotations/train')
    train_f = os.path.join(args.dataroot, 'train.txt')
    f = open(train_f, 'w')
    cnt = 0
    for annot_f in os.listdir(annot_dir):
        if os.stat(os.path.join(annot_dir, annot_f)).st_size == 0:
            label = 0
        else:
            label = 1
            cnt += 1

        img_path = os.path.join(
            args.dataroot, 'images/train', annot_f).replace('txt', 'png')
        f.write(f'{img_path},{label}')
        f.write('\n')
    f.close()
    print(f'train obj num: {cnt}')

    # val data
    annot_dir = os.path.join(args.dataroot, 'annotations/val')
    train_f = os.path.join(args.dataroot, 'val.txt')
    f = open(train_f, 'w')
    cnt = 0
    for annot_f in os.listdir(annot_dir):
        if os.stat(os.path.join(annot_dir, annot_f)).st_size == 0:
            label = 0
        else:
            label = 1
            cnt += 1

        img_path = os.path.join(
            args.dataroot, 'images/val', annot_f).replace('txt', 'png')
        f.write(f'{img_path},{label}')
        f.write('\n')
    f.close()
    print(f'val obj num: {cnt}')
