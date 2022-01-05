import os
import numpy as np
import argparse
from torchvision import transforms
from network.model_utils import get_net
import torch
from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict
import shutil
import tqdm
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/2021VRDL_HW1_datasets', help='data root dir')
parser.add_argument('--ckpt', type=str, default='/eva_data/zchin/vrdl_hw1/efficientnet_b4_1/checkpoints/best_model.pth',
                    help='checkpoint path')
parser.add_argument('--img-size', type=int, default=380, help='image size in model')
parser.add_argument('--num-classes', type=int, default=200, help='number of classes')
parser.add_argument('--net', type=str, default="efficientnet-b4", help="which model")
parser.add_argument('--gpu', type=int, default=2, help='gpu id')
parser.add_argument('--savedir',type=str,default='/eva_data/zchin/rsna_data_all/images/test_detect')
parser.add_argument('--thres',type=float,default=0.2,help='classfication probability threshold')
args = parser.parse_args()


os.makedirs(args.savedir,exist_ok=True)

def test(img_names, ckpt, img_size, net, gpu, num_classes, data_root):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = get_net(net, num_classes)
    checkpoint = torch.load(ckpt, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda(gpu)
    model.eval()
    print("model loaded...")
    print(f"epoch: {checkpoint['epoch']}")
    print(f"eval acc: {checkpoint['acc']:.4f}")

    answer = []
    with torch.no_grad():
        for img_name in tqdm.tqdm(img_names):
            img_path = os.path.join(data_root, 'images/test', img_name)
            image = Image.open(img_path).convert('RGB')
            image = data_transform(image).unsqueeze(0)

            inputs = Variable(image.cuda(gpu))
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            preds = preds.cpu().numpy()[0]

            probs=F.softmax(outputs)
            prob1=probs.cpu().squeeze(0)[1]
            
            if preds==1 or prob1>=args.thres:
                dest_path=os.path.join(args.savedir,img_name)
                shutil.copyfile(img_path,dest_path)
                answer.append([img_name])

    np.savetxt('answer.txt', answer, fmt='%s')
    print("complete...")


if __name__ == "__main__":
    with open(os.path.join(args.data_root, 'test.txt')) as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images

    # classes_f = open(os.path.join(args.data_root, 'classes.txt'), 'r', encoding='utf-8')
    # class_table = {}
    # for _class in classes_f:
    #     class_num = int(_class.strip().split('.')[0]) - 1
    #     class_table[class_num] = _class.strip()
    # print(class_table)

    test(test_images, args.ckpt, args.img_size, args.net, args.gpu, args.num_classes, args.data_root)
