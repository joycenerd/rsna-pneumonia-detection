import argparse
import os
import csv
from itertools import cycle


parser=argparse.ArgumentParser()
parser.add_argument('--out',type=str,default='./output.txt',help='CheXNet output file')
parser.add_argument('--test-img',type=str,default='/eva_data/zchin/rsna_data_all/images/test')
args=parser.parse_args()


if __name__=='__main__':
    submission=open('submission.csv','w',encoding='UTF-8',newline='\n')
    writer=csv.writer(submission)
    header=['patientId','PredictionString']
    writer.writerow(header)

    # record all the testing images
    id_dict=dict()
    for img_name in os.listdir(args.test_img):
        img_list=img_name.split('.')
        if img_list[-1]!='png':
            continue
        id_dict[img_list[0]]=0

    out=open(args.out,'r')
    lines=iter(out.readlines())
    
    while True:
        line=next(lines,-1)
        if line==-1:
            break
        info=line.strip().split(' ')
        if len(info)==2:
            _,img_name=os.path.split(info[0])
            patient_id=img_name[:-4]
            if info[1]=="1":
                pred_str=""
                next(lines,-1)
            else:
                cnt=int(info[1])-1
                next(lines,-1)
                pred_str=""
                for i in range(cnt):
                    line=next(lines,-1)
                    bbox=line.strip().split(' ')
                    if bbox[0]!='pneumonia':
                        continue
                    if i==0:
                        pred_str+=f"0.8 {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}"
                    else:
                        pred_str+=f" 0.8 {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}"
            entry=[patient_id,pred_str]
            writer.writerow(entry)
            id_dict[patient_id]=1
    
    for patient_id in id_dict:
        if id_dict[patient_id]==0:
            entry=[patient_id,""]
            writer.writerow(entry)
    
    submission.close()
    out.close()



                

