from dataset import make_dataset, Dataloader
from network.model_utils import get_net
from early_stop import EarlyStopping

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch

from pathlib import Path
import copy
import os
import argparse
import logging
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, required=True,
                    help='Your dataset root directory')
parser.add_argument('--model', type=str, default="efficientnet-b4",
                    help="which model")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--gpu', type=int, nargs='+',
                    required=True, default=[0, 1], help='gpu device')
parser.add_argument('--epochs', type=int, default=200, help='num of epoch')
parser.add_argument('--num-classes', type=int, default=200,
                    help='The number of classes for your classification problem')
parser.add_argument('--train-batch-size', type=int, default=12,
                    help='The batch size for training data')
parser.add_argument('--dev-batch-size', type=int, default=8,
                    help='The batch size for validation data')
parser.add_argument('--num-workers', type=int, default=3,
                    help='The number of worker while training')
parser.add_argument('--logs', type=str, required=True,
                    help='Directory to save all your checkpoint.pth')
parser.add_argument('--img-size', type=int, default=380,
                    help='Input image size')
parser.add_argument('--ckpt', type=str,
                    default='/eva_data/zchin/rsna_outputs/efficientnet_b4/checkpoints/epoch_30.pth', help='checkpoint path')
opt = parser.parse_args()

num_str = ",".join(str(gpu_id) for gpu_id in opt.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = num_str


if __name__ == '__main__':
    # evaluate set
    eval_set = make_dataset("eval", opt.data_root, opt.img_size)
    eval_loader = Dataloader(
        dataset=eval_set, batch_size=opt.dev_batch_size, shuffle=True, num_workers=opt.num_workers)

    # specify gpu device
    device = torch.device('cuda')

    # select model
    net = get_net(opt.model, num_classes=opt.num_classes)
    model = net
    model = nn.DataParallel(model, device_ids=opt.gpu)
    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("model loaded...")
    print(f"epoch: {checkpoint['epoch']}")
    print(f"eval acc: {checkpoint['acc']:.4f}")

    eval_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(eval_loader)):
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            eval_corrects += torch.sum(preds == labels.data)

        eval_acc = float(eval_corrects) / len(eval_set)

        print(f'val acc: {eval_acc}')
