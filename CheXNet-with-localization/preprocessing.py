import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from os import listdir
import skimage.transform
import pickle
import sys, os
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer,OneHotEncoder
import argparse
import glob
import cv2
from tqdm import tqdm

# image_folder_path = sys.argv[1] # folder contain all images
# data_entry_path = sys.argv[2] 
# bbox_list_path = sys.argv[3]
# train_txt_path = sys.argv[4]
# valid_txt_path = sys.argv[5]
# data_path = sys.argv[6] # ouput folder for preprocessed data

parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/rsna_data_all',help='data root dir')
args=parser.parse_args()


train_dir=os.path.join(args.dataroot,'images/train')
train_list=[]
for img_name in os.listdir(train_dir):
    train_list.append(img_name)

# transform training images
print("training example:",len(train_list))
print("take care of your RAM here !!!")
train_X = []
for img_name in tqdm(train_list):
    image_path = os.path.join(train_dir,img_name)
    img = imageio.imread(image_path)
    img=cv2.equalizeHist(img)
    if img.shape != (1024,1024): # there some image with shape (1024,1024,4) in training set
        img = img[:,:,0]
    img_resized = skimage.transform.resize(img,(256,256)) # or use img[::4] here
    train_X.append((np.array(img_resized)/255).reshape(256,256,1))
    # if i % 3000==0:
    #     print(i)

train_X = np.array(train_X)
np.save(os.path.join(args.dataroot,"train_X_small.npy"), train_X)

valid_dir=os.path.join(args.dataroot,'images/val')
valid_list=[]
for img_name in os.listdir(valid_dir):
    valid_list.append(img_name)

# transform validation images
print("validation example:",len(valid_list))
valid_X = []
for img_name in tqdm(valid_list):
    image_path = os.path.join(valid_dir,img_name)
    img = imageio.imread(image_path)
    img=cv2.equalizeHist(img)
    if img.shape != (1024,1024):
        img = img[:,:,0]
    img_resized = skimage.transform.resize(img,(256,256))
    valid_X.append((np.array(img_resized)/255).reshape(256,256,1))
    # if i % 3000==0:
    #     print(i)

valid_X = np.array(valid_X)
np.save(os.path.join(args.dataroot,"valid_X_small.npy"), valid_X)


# process label
print("label preprocessing")

train_y = []
train_annot_dir=os.path.join(args.dataroot,'annotations/train')
for train_id in train_list:
    annot_f=os.path.join(train_annot_dir,train_id.replace('png','txt'))
    if os.stat(annot_f).st_size==0:
        train_y.append('normal')
    else:
        train_y.append('pneumonia')

valid_y = []
valid_annot_dir=os.path.join(args.dataroot,'annotations/val')
for valid_id in valid_list:
    annot_f=os.path.join(valid_annot_dir,valid_id.replace('png','txt'))
    if os.stat(annot_f).st_size==0:
        valid_y.append('normal')
    else:
        valid_y.append('pneumonia')


encoder = LabelBinarizer()
# encoder.fit(train_y+valid_y)
train_y_onehot = encoder.fit_transform(train_y)
train_y_onehot = np.hstack((train_y_onehot, 1 - train_y_onehot))
valid_y_onehot = encoder.fit_transform(valid_y)
valid_y_onehot = np.hstack((valid_y_onehot, 1 - valid_y_onehot))

# train_y_onehot = np.delete(train_y_onehot, [2,3,5,6,7,10,12],1) # delete out 8 and "No Finding" column
# valid_y_onehot = np.delete(valid_y_onehot, [2,3,5,6,7,10,12],1) # delete out 8 and "No Finding" column

with open(args.dataroot + "/train_y_onehot.pkl","wb") as f:
    pickle.dump(train_y_onehot, f)
with open(args.dataroot + "/valid_y_onehot.pkl","wb") as f:
    pickle.dump(valid_y_onehot, f)
with open(args.dataroot + "/label_encoder.pkl","wb") as f:
    pickle.dump(encoder, f)
