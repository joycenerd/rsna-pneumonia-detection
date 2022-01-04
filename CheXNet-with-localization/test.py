import numpy as np
from os import listdir
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches
import argparse
import glob
import imageio
import tqdm
import shutil


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
test_X = []


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/rsna_data_all/images/test',help='test image save dir')
parser.add_argument('--weights',type=str,default='/eva_data/zchin/rsna_outputs/CheXNet/DenseNet121_aug2_pretrain_noWeight_2_0.8875203559299248.pkl',help='trained model weights')
parser.add_argument('--savedir',type=str,default='/eva_data/zchin/rsna_data_all/images/test_detect',help='output of classification')
args=parser.parse_args()



def get_file_names(dataroot):
    imgs=[]
    for img_path in glob.glob(dataroot+'/*'):
        imgs.append(img_path)
    return imgs


# imgs = judger.get_file_names()
# f = judger.get_output_file_object()
imgs=get_file_names(args.dataroot)

for img in tqdm.tqdm(imgs):
    # img = scipy.misc.imread(img)
    img=imageio.imread(img)
    if img.shape != (1024,1024):
        img = img[:,:,0]
    img_resized = skimage.transform.resize(img,(256,256))
    test_X.append((np.array(img_resized)).reshape(256,256,1))
test_X = np.array(test_X)
print(test_X.shape)


# model archi
# construct model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

model = DenseNet121(2).cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.weights))
print("model loaded")


# build test dataset
class ChestXrayDataSet_plot(Dataset):
    def __init__(self, input_X = test_X, transform=None):
        self.X = np.uint8(test_X*255)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image 
        """
        current_X = np.tile(self.X[index],3)
        image = self.transform(current_X)
        return image
    def __len__(self):
        return len(self.X)

test_dataset = ChestXrayDataSet_plot(input_X = test_X,transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]))
os.makedirs(args.savedir,exist_ok=True)

# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
#         self.probs = F.softmax(self.preds)[0]
#         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().data.numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data
        
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


# ======== Create heatmap ===========

heatmap_output = []
image_id = []
output_class = []

gcam = GradCAM(model=model, cuda=True)
for index in tqdm.tqdm(range(len(test_dataset))):
    input_img = Variable((test_dataset[index]).unsqueeze(0).cuda(), requires_grad=True)
    probs = gcam.forward(input_img)
    probs=probs[0]
    if probs[0]>probs[1]:
        heatmap_output.append(np.full((224,224),np.nan))
        image_id.append(index)
        output_class.append(0)
        gcam.backward(idx=0)
    else:
        _,img_name=os.path.split(imgs[index])
        dest_path=os.path.join(args.savedir,img_name)
        shutil.copyfile(imgs[index],dest_path)
        