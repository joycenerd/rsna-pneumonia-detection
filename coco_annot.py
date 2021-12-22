import json
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/rsna_data',help='data root path')
parser.add_argument('--mode',type=str,default='train',help='generate train or val annotation')
args=parser.parse_args()

def create_img_annot(img_path,annot_path,img_id,_id):
    # create image
    image=dict()
    image['id']=img_id
    im=cv2.imread(img_path)
    h,w,_=im.shape
    image['width']=w
    image['height']=h
    image['file_name']=img_path.split('/')[-1]

    # create annotation
    annotation=[]
    annot_f=open(annot_path,'r')
    for row in annot_f:
        info=row.strip('\n').split(' ')
        annot=dict()
        annot['id']=_id
        annot['image_id']=img_id
        annot['category_id']=0
        
        # process bbox (yolo->coco)
        x_center,y_center,w_norm,h_norm=float(info[1]),float(info[2]),float(info[3]),float(info[4])
        width=w_norm*w
        height=h_norm*h
        x_min=x_center*w-width/2
        y_min=y_center*h-height/2
        annot['bbox']=[x_min,y_min,width,height]
        annot['area']=width*height

        annot['iscrowd']=0

        annotation.append(annot)
        _id+=1

    return image,annotation,_id


if __name__=='__main__':
    annot_dict=dict()
    images=[]
    annotations=[]
    categories=[]

    # create category
    category=dict()
    category['id']=0
    category['name']='pneumonia'
    category['supercategory']=None
    categories.append(category)

    img_dir=os.path.join(args.dataroot,f'images/{args.mode}')
    annot_dir=os.path.join(args.dataroot,f'annotations/{args.mode}')
    
    img_id=0
    _id=1

    pbar=tqdm(os.listdir(img_dir))
    for img in pbar:
        pbar.set_description(img)

        # add image and annotation
        img_path=os.path.join(img_dir,img)
        annot_path=os.path.join(annot_dir,img.replace('png','txt'))
        image,annotation,_id=create_img_annot(img_path,annot_path,img_id,_id)
        images.append(image)
        annotations.extend(annotation)
        img_id+=1

    # dump to json
    annot_dict['images']=images
    annot_dict['categories']=categories
    annot_dict['annotations']=annotations

    annot_path=os.path.join(args.dataroot,f'annotations/instance_{args.mode}.json')
    annot_f=open(annot_path,'w')
    json.dump(annot_dict,annot_f,indent=4)
    annot_f.close()
