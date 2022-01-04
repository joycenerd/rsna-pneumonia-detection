import os
import numpy as np
import argparse
from torchvision import transforms
from network.model_utils import get_net
import torch
from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/eva_data/zchin/2021VRDL_HW1_datasets', help='data root dir')
parser.add_argument('--ckpt', type=str, default='./checkpoint/efficientnet-b4_best_model.pth', help='checkpoint path')
parser.add_argument('--img-size', type=int, default=380, help='image size in model')
parser.add_argument('--num-classes', type=int, default=200, help='number of classes')
parser.add_argument('--net', type=str, default="efficientnet-b4", help="which model")
parser.add_argument('--gpu', type=int, default=2, help='gpu id')
args = parser.parse_args()


# submission = []
# for img in test_images:  # image order is important to your result
#     predicted_class = your_model(img)  # the predicted category
#     submission.append([img, predicted_class])

# np.savetxt('answer.txt', submission, fmt='%s')

def test(img_names, ckpt, img_size, net, gpu, num_classes, data_root, class_table):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load model and let it run on one gpu
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

    submission = []
    for img_name in img_names:
        img_path = os.path.join(data_root, 'testing_images', img_name)
        image = Image.open(img_path).convert('RGB')
        image = data_transform(image).unsqueeze(0)

        inputs = Variable(image.cuda(gpu))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy()[0]
        submission.append([img_name, class_table[preds]])

    np.savetxt('answer.txt', submission, fmt='%s')
    print("complete...")


if __name__ == "__main__":
    with open(os.path.join(args.data_root, 'testing_img_order.txt')) as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images

    classes_f = open(os.path.join(args.data_root, 'classes.txt'), 'r', encoding='utf-8')
    class_table = {}
    for _class in classes_f:
        class_num = int(_class.strip().split('.')[0]) - 1
        class_table[class_num] = _class.strip()
    # print(class_table)

    test(test_images, args.ckpt, args.img_size, args.net, args.gpu, args.num_classes, args.data_root, class_table)
