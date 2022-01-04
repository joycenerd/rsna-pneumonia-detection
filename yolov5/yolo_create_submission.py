import json
import argparse
import csv
import os


parser=argparse.ArgumentParser()
parser.add_argument('--pred-json',type=str,default='/eva_data/zchin/rsna_outputs/yolov5s/test/exp2/best_predictions.json',help='original prediction output file')
parser.add_argument('--test-img',type=str,default='/eva_data/zchin/rsna_data/images/test')
args=parser.parse_args()


if __name__=='__main__':
    # load original prediction file
    f=open(args.pred_json)
    data=json.load(f)
    
    # create submission file
    out_f=open('submission.csv','w',encoding='UTF-8',newline='\n')
    writer=csv.writer(out_f)
    header=['patientId','PredictionString']
    writer.writerow(header)

    # record all the testing images
    id_dict=dict()
    for img_name in os.listdir(args.test_img):
        img_list=img_name.split('.')
        if img_list[-1]!='png':
            continue
        id_dict[img_list[0]]=0

    # write result to submission file
    prev_id=None
    pred_str=""
    cnt=0
    for pred in data:
        patient_id=pred['image_id']
        conf=pred['score']
        x_min,y_min,width,height=pred['bbox'][0],pred['bbox'][1],pred['bbox'][2],pred['bbox'][3]
        # width*=0.875
        # height*=0.875

        if patient_id==prev_id:
            pred_str+=f' {conf} {x_min} {y_min} {width} {height}'
        else:
            if prev_id!=None:
                entry=[prev_id, pred_str]
                writer.writerow(entry)
            pred_str=f'{conf} {x_min} {y_min} {width} {height}'
            id_dict[patient_id]=1
            prev_id=patient_id
    entry=[patient_id,pred_str]
    writer.writerow(entry)

    for patient_id in id_dict:
        if id_dict[patient_id]==0:
            entry=[patient_id,""]
            writer.writerow(entry)
    
    f.close()
    out_f.close()


