import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
from retinanet.model import resnet101

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path):

    classes= {0: u'Lung Opacity'}
    #1: u'person',
    #2: u'bicycle'

    labels = {}
    for key, value in classes.items():
        labels[value] = key
    #model = resnet18(num_classes=2)
    #model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    #model = torch.load(model_path,map_location=torch.device('cpu'))

    # retinanet = resnet101(num_classes=2, pretrained=True)
    
    # retinanet.load_state_dict(torch.load("csv_retinanet_8.pt",map_location=torch.device('cpu')))
    # model = retinanet
    model = torch.load(model_path,map_location=torch.device('cpu'))
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    out_f=open('test.csv','w',encoding='UTF-8',newline='\n')
    writer=csv.writer(out_f)
    header=['patientId','PredictionString']
    writer.writerow(header)
    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            scores, classification, transformed_anchors = model(image.cuda().float())
            idxs = np.where(scores.cpu() > 0.25)
            pred_str = ""
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                #label_name = labels[int(classification[idxs[0][j]])]
                label_name = str(int(classification[idxs[0][j]]))
                score = scores[j]
                if x2-x1 < 800 and y2-y1 < 800:
                    pred_str+=f' {score} {x1} {y1} {(x2-x1)*0.875} {(y2-y1)*0.875}'
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            entry=[img_name.split(".")[0],pred_str]
            writer.writerow(entry)
            # cv2.imshow('detections', image_orig)
            # cv2.waitKey(0)
            # cv2.imwrite("retina_result/"+img_name,image_orig)
        print(f'finish {img_name}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path)