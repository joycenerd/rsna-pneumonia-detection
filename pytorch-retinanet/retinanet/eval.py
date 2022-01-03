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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
from log import *

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

LOG_SIZE = 512 * 1024 * 1024 # 512M
LOGGER_NAME = 'eval'
LOG_PATH = './log'

# scan score thresholds
SCORE_THRESHOLDS = [0.01, 0.03, 0.05, 0.07, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
MAX_DETECTIONS = 3

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for evaluating a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

	parser.add_argument('--epochs', help='Number of epochs', type=int, default=0)
	parser.add_argument('--checkpoint', help='Path to checkpoint')

	parser.add_argument('--ensemble', action='store_true', default=False, help='Whether to do ensemble')
	parser.add_argument('--ensemble_file', default='./ensemble_list.txt', help='checkpoint list file for ensemble')

	parser.add_argument('--log_prefix', default='eval', help='log file path = "./log/{}-{}.log".format(log_prefix, now)')
	parser.add_argument('--log_level', default=logging.DEBUG, type=int, help='log level')

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
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

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
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()
	retinanet.training = False

	if parser.epochs > 0:
		for epoch in range(parser.epochs):
			logger.info('Epoch {}:'.format(epoch))

			model_file = '{}_{}.pth'.format(parser.checkpoint, epoch)
			print('Evaluating model: {}'.format(model_file))

			retinanet.load_state_dict(torch.load(model_file), True)

			ap_list, youden_list, sensitivity_list, specificity_list = csv_eval.evaluate_rsna(
				dataset_val,
				retinanet,
				score_thresholds=SCORE_THRESHOLDS,
				max_detections=MAX_DETECTIONS
			)
			logger.info(
				'\nscores:\t\t{}\nmAPs:\t\t{}\nyouden:\t\t{}\nsensitivity:\t{}\nspecificity:\t{}\n'.format(
					SCORE_THRESHOLDS,
					ap_list,
					youden_list,
					sensitivity_list,
					specificity_list
				)
			)

			print(
				'\nscores:\t\t{}\nmAPs:\t\t{}\nyouden:\t\t{}\nsensitivity:\t{}\nspecificity:\t{}\n'.format(
					SCORE_THRESHOLDS,
					ap_list,
					youden_list,
					sensitivity_list,
					specificity_list
				)
			)
	else:
		if parser.ensemble:
			# read ensemble list file
			with open(parser.ensemble_file, 'r') as ensemble_file:
				ensemble_list = ensemble_file.readlines()
				ensemble_list = [filename.strip() for filename in ensemble_list]
			
			logger.info('Ensembling {} checkpoints:'.format(len(ensemble_list)))
			for filename in ensemble_list:
				logger.info('{}'.format(filename))

			print('Ensembling {} checkpoints...'.format(len(ensemble_list)))

			retinanet = model.resnet101_ensemble(
				num_classes=dataset_val.num_classes(),
				checkpoint_list=ensemble_list
			)
		else:
			retinanet.load_state_dict(torch.load(parser.checkpoint), False)

		ap_list, youden_list, sensitivity_list, specificity_list = csv_eval.evaluate_rsna(
			dataset_val,
			retinanet,
			score_thresholds=SCORE_THRESHOLDS,
			max_detections=MAX_DETECTIONS
		)
		logger.info(
			'\nscores:\t\t{}\nmAPs:\t\t{}\nyouden:\t\t{}\nsensitivity:\t{}\nspecificity:\t{}\n'.format(
				SCORE_THRESHOLDS,
				ap_list,
				youden_list,
				sensitivity_list,
				specificity_list
			)
		)

		print(
			'\nscores:\t\t{}\nmAPs:\t\t{}\nyouden:\t\t{}\nsensitivity:\t{}\nspecificity:\t{}\n'.format(
				SCORE_THRESHOLDS,
				ap_list,
				youden_list,
				sensitivity_list,
				specificity_list
			)
		)

if __name__ == '__main__':
	main()