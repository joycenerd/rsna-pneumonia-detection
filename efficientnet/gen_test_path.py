import os
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--dataroot',type=str,default='/eva_data/zchin/rsna_data_all',help='data root dir')
args=parser.parse_args()


if __name__=='__main__':
    test_dir=os.path.join(args.dataroot,'images/test')
    with open(os.path.join(args.dataroot,'test.txt'),'w') as f:
        for path in os.listdir(test_dir):
            f.write(path)
            f.write('\n')
    
