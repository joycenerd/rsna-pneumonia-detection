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
parser.add_argument('--data-root', type=str, required=True, help='Your dataset root directory')
parser.add_argument('--model', type=str, default="efficientnet-b4",
                    help="which model")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--gpu', type=int, nargs='+', required=True, default=[0, 1], help='gpu device')
parser.add_argument('--epochs', type=int, default=200, help='num of epoch')
parser.add_argument('--num-classes', type=int, default=200,
                    help='The number of classes for your classification problem')
parser.add_argument('--train-batch-size', type=int, default=12,
                    help='The batch size for training data')
parser.add_argument('--dev-batch-size', type=int, default=8,
                    help='The batch size for validation data')
parser.add_argument('--num-workers', type=int, default=3,
                    help='The number of worker while training')
parser.add_argument('--logs', type=str, required=True, help='Directory to save all your checkpoint.pth')
parser.add_argument('--img-size', type=int, default=380,
                    help='Input image size')
opt = parser.parse_args()

num_str = ",".join(str(gpu_id) for gpu_id in opt.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = num_str


def train():
    # train set
    train_set = make_dataset("train", opt.data_root, opt.img_size)
    train_loader = Dataloader(
        dataset=train_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers)

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
    model.to(device)

    best_acc = 0.0

    # initialize optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True, cooldown=1)
    early_stopping = EarlyStopping(patience=15, verbose=True)

    for epoch in range(opt.epochs):
        log_string(f'Epoch: {epoch + 1}/{opt.epochs}')

        training_loss = 0.0
        training_corrects = 0

        model.train()

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = float(training_corrects) / len(train_set)

        writer.add_scalar("train loss/epochs", training_loss, epoch + 1)
        writer.add_scalar("train accuracy/epochs", training_acc, epoch + 1)

        log_string(f'Train loss: {training_loss:.4f}\tacc: {training_acc:.4f}')

        model.eval()

        eval_loss = 0.0
        eval_corrects = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(eval_loader)):
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                eval_loss += loss.item() * inputs.size(0)
                eval_corrects += torch.sum(preds == labels.data)

        eval_loss = eval_loss / len(eval_set)
        eval_acc = float(eval_corrects) / len(eval_set)

        writer.add_scalar("eval loss/epochs", eval_loss, epoch + 1)
        writer.add_scalar("eval accuracy/epochs", eval_acc, epoch + 1)
        log_string(f'Eval loss: {eval_loss:.4f}\taccuracy: {eval_acc:.4f}')
        
        scheduler.step(eval_loss)
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                'acc': eval_acc
            }, os.path.join(ckpt_dir, 'best_model.pth'))

        if (epoch + 1) % 10 == 0:
        # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                'acc': eval_acc
            }, os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pth'))

        log_string(f'Best acc: {best_acc}')


def set_logger(log_dir):
    # Setup LOG file format
    global logger
    logger = logging.getLogger(opt.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, opt.model + ".txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def log_string(message):
    # Write message into log.txt
    logger.info(message)
    print(message)


if __name__ == "__main__":
    ckpt_dir = os.path.join(opt.logs, 'checkpoints')
    event_dir = os.path.join(opt.logs, 'events')
    log_dir = os.path.join(opt.logs, 'logger')

    if not os.path.isdir(opt.logs):
        os.makedirs(opt.logs)
        os.mkdir(ckpt_dir)
        os.mkdir(event_dir)
        os.mkdir(log_dir)

    set_logger(log_dir)
    log_string(opt)
    writer = SummaryWriter(event_dir)

    train()
