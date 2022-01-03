import time
import os
import copy
import argparse
import pdb
import collections
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import retinanet.model
from retinanet.anchors import Anchors
import retinanet.losses
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import retinanet.coco_eval
import retinanet.csv_eval
from log import *

assert torch.__version__.split('.')[1] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

LOG_SIZE = 512 * 1024 * 1024 # 512M
LOGGER_NAME = 'detect'
LOG_PATH = './log'

SCORE_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
MAX_DETECTIONS = 3

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for evaluating a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--image_size', help='Image size', type=int, default=608)
	
	parser.add_argument('--scale', help='Resize scale', type=float, default=0.9)
	
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--checkpoint', help='Path to checkpoint')

	parser.add_argument('--ensemble', action='store_true', default=False, help='Whether to do ensemble')
	parser.add_argument('--ensemble_file', default='./ensemble_list.txt', help='checkpoint list file for ensemble')

	parser.add_argument('--log_prefix', default='eval', help='log file path = "./log/{}-{}.log".format(log_prefix, now)')
	parser.add_argument('--log_level', default=logging.DEBUG, type=int, help='log level')

	parser.add_argument('--tag', default='resnet', help='custom tag for submission filename')

	parser = parser.parse_args(args)

	# setup logger
	if not os.path.isdir(LOG_PATH):
		os.mkdir(LOG_PATH)

	now = datetime.now()

	logger = setup_logger(
		LOGGER_NAME,
		os.path.join(
			LOG_PATH,
			'{}_{}.log'.format(parser.log_prefix, now.strftime('%Y-%m-%d_%H:%M:%S'))
		),
		LOG_SIZE,
		parser.log_level
	)

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on CSV,')

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer(size=parser.image_size)]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=8, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True, global_flag=False)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True, global_flag=False)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True, global_flag=False)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True, global_flag=False)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True, global_flag=False)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')	

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	if parser.ensemble:
		# read ensemble list file
		ensemble_df = pd.read_csv(parser.ensemble_file)
		
		logger.info('Ensembling {} checkpoints:'.format(len(ensemble_df)))
		for i, row in ensemble_df.iterrows():
			logger.info('{}'.format(row['filename']))

		print('Ensembling {} checkpoints...'.format(len(ensemble_df)))

		retinanet = model.resnet_ensemble(
			num_classes=dataset_val.num_classes(),
			checkpoint_list=ensemble_df
		)
		ensemble_score = retinanet.score_threshold
		SCORE_THRESHOLDS.append(ensemble_score)

		print('Ensemble score threshold: {}, weights: {}'.format(ensemble_score, retinanet.weights))
	else:
		retinanet = torch.nn.DataParallel(retinanet).cuda()
		retinanet.training = False
		
		retinanet.load_state_dict(torch.load(parser.checkpoint), False)

	csv_eval.export(
		dataset_val,
		retinanet,
		score_thresholds=SCORE_THRESHOLDS,
		max_detections=MAX_DETECTIONS,
		csv_path='./submission_{}_{}_{}.csv',
		scale=parser.scale,
		tag=parser.tag
	)

if __name__ == '__main__':
	main()