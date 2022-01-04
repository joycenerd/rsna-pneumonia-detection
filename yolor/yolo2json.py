import os
import cv2
import json
import operator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--txt-path', type=str, help='output file of prediction')
parser.add_argument('--test-img', type=str, help='path to test image')
args = parser.parse_args()


txt_path = args.txt_path
img_path = args.test_img
entries = os.listdir(img_path)
data = []
for i in range(len(entries)):
    img_name = entries[i].split(".")[0]
    if not os.path.isfile(txt_path+entries[i].replace('.png', '.txt')):
        a = {"image_id": img_name,
             "bbox": (1, 1, 1, 1), "score": 0.5, "label": 0}
    else:
        f = open(txt_path+entries[i].replace('.png', '.txt'), 'r')
        contents = f.readlines()
        im = cv2.imread(img_path + '/' + img_name + '.png')
        h, w, c = im.shape
        for content in contents:
            a = dict.fromkeys(['image_id', 'bbox', 'score'])
            content = content.replace('\n', '')
            c = content.split(' ')
            a['image_id'] = img_name
            w_center = w*float(c[1])
            h_center = h*float(c[2])
            width = w*float(c[3])
            height = h*float(c[4])
            left = float(w_center - width/2)
            top = float(h_center - height/2)
            a['bbox'] = (tuple((left, top, width, height)))
            a['score'] = (float(c[5]))
            # a['category_id'] = (int(c[0]))
            data.append(a)
        f.close()
json_object = json.dumps(data, indent=4)

print(len(data))
with open('answer.json', 'w') as fp:
    fp.write(json_object)
