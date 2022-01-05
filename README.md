# rsna-pneumonia-detection
Final project of VRDL course in 2021 fall semester at NYCU. 

# YOLOR

## Clone the repo
```
git clone https://github.com/joycenerd/rsna-pneumonia-detection.git
```

## Prepare data
```
mkdir rsna_data && cd rsna_data
download the data from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
unzip rsna-pneumonia-detection-challenge.zip
cd ../rsna-pneumonia-detection
```

## Setup the environment
```
conda env create -f yolor.yml
conda activate rsna_test
```
## Convert dcm to png
```
python dcm2png.py --dataroot ../rsna_data/ --mode train
python dcm2png.py --dataroot ../rsna_data/ --mode test
```

## Create yolo annotation
```
python yolo_annot.py --dataroot ../rsna_data/ --destroot ../rsna_data/
```

## Split train & valid data
```
python train_valid_split.py --data-root ../rsna_data/ --ratio 0.1
```

## Training yolor
Change the filepath in yolor/config.yaml
```
cd yolor
python train.py --batch-size 4 --img 1280 1280 --data config.yaml --cfg cfg/yolor_w6.cfg --weights '' --device 3 --name yolor_w6 --hyp hyp.scratch.1280.yaml --epochs 300
```

## Download the checkpoint
https://drive.google.com/file/d/1pNnBwk_KVDgIavoRxRkF5hbMxWw7xOw3/view?usp=sharing
## Inference 
```
python detect.py --source <path_to_testing_images> --cfg cfg/yolor_w6.cfg --weights best.pt --conf 0.234 --img-size 1024 --device 0 --save-txt --output rsna_output/yolor_w6
```
## Create prediction json file
```
python yolo2json.py --txt-path rsna_output/yolor_w6/ --test-img ../../rsna_data/images/test/
```
## Create csv submission
```
python yolo_create_submission.py --pred-json answer.json --test-img ../../rsna_data/images/test
```
