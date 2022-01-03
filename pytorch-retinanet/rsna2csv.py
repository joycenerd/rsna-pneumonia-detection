# -*- coding: utf-8 -*-
import sys
import os
import pickle
import argparse
import itertools
import math
import random
import shutil

import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification

# constants
RSNA_LABEL_FILE = 'stage_2_train_labels.csv'
RSNA_CLASS_FILE = 'stage_2_detailed_class_info.csv'
global_class_mapping = {
	'Normal': 0,
	'No Lung Opacity / Not Normal': 1,
	'Lung Opacity': 2
}
CLASS_MAPPING = {
    'Lung Opacity': 0
}
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# helpers
def to_point(corner):
    return [
        corner[0],
        corner[1],
        corner[0] + corner[2],
        corner[1] + corner[3]
    ]

def convert_jpeg(filename, patient_id, output_path):
    output_filename = os.path.join(output_path, '{}.jpg'.format(patient_id))

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    if not os.path.exists(output_filename):
        ds = sitk.ReadImage(filename)
        img_array = sitk.GetArrayFromImage(ds)

        cv2.imwrite(
            output_filename,
            img_array[0]
        )

def convert(filename, sample_list, anno_df, class_df, root_path, classification_flag):
    lines = []
    
    print('Export {}...'.format(filename))

    for patient_id in tqdm(sample_list):
        class_rows = class_df[class_df['patientId'] == patient_id]
        class_row = class_rows.iloc[0]
        class_name = class_row['class']

        anno_rows = anno_df[anno_df['patientId'] == patient_id]
        image_path = os.path.join(root_path, '{}.dcm'.format(patient_id))
        jpg_path = os.path.join('{}_jpg'.format(root_path), '{}.jpg'.format(patient_id))

        # convert to jpeg
        convert_jpeg(
            image_path,
            patient_id,
            '{}_jpg'.format(root_path)
        )

        for index, anno_row in anno_rows.iterrows():
            if class_name == 'Normal':
                # write an empty row
                line = '{},,,,,\n'.format(jpg_path)
            elif class_name == 'No Lung Opacity / Not Normal':
                if classification_flag:
                    # write a whole image annotation
                    line = '{},{},{},{},{},{}\n'.format(
                        jpg_path,
                        0,
                        0,
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                        class_name
                    )
                else:
                    # write an empty line
                    line = '{},,,,,\n'.format(jpg_path)
            else: # Lung Opacity
                # convert coords
                a_cn = [
                    anno_row['x'],
                    anno_row['y'],
                    anno_row['width'],
                    anno_row['height']
                ]

                a_pt = to_point(a_cn)

                # write bbox row
                line = '{},{},{},{},{},{}\n'.format(
                    jpg_path,
                    int(a_pt[0]),
                    int(a_pt[1]),
                    int(a_pt[2]),
                    int(a_pt[3]),
                    class_name
                )
            
            lines.append(line)

    # write csv
    csv_path = os.path.join('./', filename)

    with open(csv_path, 'w') as file:
        for line in lines:
            file.write(line)

def convert_test(filename, sample_list, anno_df, class_df, root_path, classification_flag):
    lines = []
    
    print('Export {}...'.format(filename))

    for patient_id in tqdm(sample_list):
        image_path = os.path.join(root_path, '{}.dcm'.format(patient_id))
        jpg_path = os.path.join('{}_jpg'.format(root_path), '{}.jpg'.format(patient_id))

        # convert to jpeg
        convert_jpeg(
            image_path,
            patient_id,
            '{}_jpg'.format(root_path)
        )

        line = '{},,,,,\n'.format(jpg_path)
        lines.append(line)

    # write csv
    csv_path = os.path.join('./', filename)

    with open(csv_path, 'w') as file:
        for line in lines:
            file.write(line)

# argparser
parser = argparse.ArgumentParser(description='RSNA dataset convertor (to CSV dataset)')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--tag', default='rsna', help='output subpath')
parser.add_argument('--fold', default=4, type=int, help='sub-sets of k-fold for training, set 1 to disable k-fold')
parser.add_argument('--val', default=0.05, type=float, help='ratio for online validation if k-fold is disabled')
parser.add_argument('--classification', default=False, action='store_true', help='whether global classification is added to network (deprecated)')
flags = parser.parse_args()

if __name__ == '__main__':
    if not os.path.isdir('./{}'.format(flags.tag)):
        os.mkdir('./{}'.format(flags.tag))

    anno_path = os.path.join(flags.root, RSNA_LABEL_FILE)
    class_path = os.path.join(flags.root, RSNA_CLASS_FILE)

    train_path = os.path.join(flags.root, 'stage_2_train_images')
    test_path = os.path.join(flags.root, 'stage_2_test_images')

    anno_df = pd.read_csv(anno_path)
    class_df = pd.read_csv(class_path)

    print('Getting unique patient class from {}...'.format(RSNA_CLASS_FILE))
    
    patient_list = set(class_df['patientId'])
    patient_list = list(patient_list)
    class_list = [(class_df[class_df['patientId'] == patientId]).iloc[0]['class'] for patientId in patient_list]

    class_list = [[global_class_mapping[class_name]] for class_name in class_list]
    class_list = np.array(class_list)

    if flags.fold == 1:
        patient_train_indexes, _, patient_val_indexes, _ = iterative_train_test_split(
            patient_list,
            class_list,
            test_size = flags.val
        )

        convert(
            os.path.join('{}'.format(flags.tag), 'rsna-train.csv'),
            [patient_list[i] for i in patient_train_indexes],
            anno_df,
            class_df,
            train_path,
            flags.classification
        )

        convert(
            os.path.join('{}'.format(flags.tag), 'rsna-val.csv'),
            [patient_list[i] for i in patient_val_indexes],
            anno_df,
            class_df,
            train_path,
            flags.classification
        )
    else:
        # create stratified lists
        k_fold = IterativeStratification(n_splits=flags.fold, order=1)

        train_sample_lists = []
        val_sample_lists = []

        for train_index_list, val_index_list in k_fold.split(patient_list, class_list):
            train_sample_list = [patient_list[i] for i in train_index_list]
            val_sample_list = [patient_list[i] for i in val_index_list]

            train_sample_lists.append(train_sample_list)
            val_sample_lists.append(val_sample_list)

        # write train and val sets
        for i in range(flags.fold):
            convert(
                os.path.join('{}'.format(flags.tag), 'rsna-train-{}.csv'.format(i)),
                train_sample_lists[i],
                anno_df,
                class_df,
                train_path,
                flags.classification
            )

            convert(
                os.path.join('{}'.format(flags.tag), 'rsna-val-{}.csv'.format(i)),
                val_sample_lists[i],
                anno_df,
                class_df,
                train_path,
                flags.classification
            )

    # test
    if os.path.exists(test_path):
        test_list = os.listdir(test_path)
        test_list = [filename.split('.')[0] for filename in test_list]

        convert_test(
            os.path.join('{}'.format(flags.tag), 'rsna-test.csv'),
            test_list,
            anno_df,
            class_df,
            test_path,
            flags.classification
        )
        
    # class mapping
    with open(os.path.join('./{}'.format(flags.tag), 'rsna-class-mapping.csv'), 'w') as file:
        for key in CLASS_MAPPING.keys():
            file.write('{},{}\n'.format(key, CLASS_MAPPING[key]))