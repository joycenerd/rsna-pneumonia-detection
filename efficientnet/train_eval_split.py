import os
import math
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/2021VRDL_HW1_datasets',
                    help='raw data saving path')
args = parser.parse_args()

data_root = args.data_root

train_label_f = open(os.path.join(data_root, 'training_labels.txt'), 'r', encoding='utf-8')
entries = []
for line in train_label_f:
    entries.append(line.strip())

eval_size = math.floor(len(entries) * 0.3)
idx = random.sample(range(len(entries)), eval_size)

new_train = open(os.path.join(data_root, 'new_train_label.txt'), 'w')
new_eval = open(os.path.join(data_root, 'new_eval_label.txt'), 'w')

for i in range(len(entries)):
    line_list = entries[i].split()
    img_name, raw_label = line_list[0], line_list[1]
    img_path = os.path.join(data_root, 'training_images', img_name)
    label = int(raw_label[:3]) - 1
    if i in idx:
        new_eval.write(f'{img_path},{label}')
        new_eval.write('\n')
    else:
        new_train.write(f'{img_path},{label}')
        new_train.write('\n')
