# Bird Images Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### [Report](./REPORT.pdf)

by [Zhi-Yi Chin](https://joycenerd.github.io/)

This repository is implementation of homework1 for IOC5008 Selected Topics in Visual Recognition using Deep Learning course in 2021 fall semester at National Yang Ming Chiao Tung University.

In this homework, we participated a bird image classification challenge hosted on [CodaLab](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07). This challenge provided a total of 6,033 bird images (3000 for training and 3033 for testing) belonging to 200 bird species. For the fairness of this challenge, external data is not allowed to train our model. We can use pre-trained models that have been released by other people online, but is limited to pre-train on [ImageNet](https://www.image-net.org/).

In data-preprocessing, we applied many data augmentations: RandomAffine, RandomGrayscale, RandomHorizontalFlip, RandomPerspective, RandomVerticalFlip, ColorJitter, RandomShift, color transform, RandomRotation. We applied EfficientNet-b4 as our classification model.

## Getting the code

You can download a copy of all the files in this repository by cloning this repository:

```
git clone https://github.com/joycenerd/bird-images-classification.git
```

## Requirements

You need to have [Anaconda](https://www.anaconda.com/) or Miniconda already installed in your environment. To install requirements:
```
conda env create -f environment.yml
conda activate classify
```

## Dataset

You can download the raw data after you have registered the challenge mention above. You can get my training and evaluation data split by moving two text files in `data/` into your downloaded raw data directory. Or you can generate these two text files by yourself by running this command:
```
python train_eval_split.py --data-root <path_to_data>
```
There will be two files generated in your raw data directory:
* `new_train_label.txt`: record the training images path and labels
* `new_eval_label.txt`: record the validation images path and labels

## Training

You should have Graphics card to train the model. For your reference, we trained on 2 NVIDIA RTX 1080Ti. To train the model, run this command:
```
python train.py [-h] --data-root DATA_ROOT [--model MODEL] [--lr LR] --gpu GPU
                [GPU ...] [--epochs EPOCHS] [--num-classes NUM_CLASSES]
                [--train-batch-size TRAIN_BATCH_SIZE]
                [--dev-batch-size DEV_BATCH_SIZE] [--num-workers NUM_WORKERS] --logs
                LOGS [--img-size IMG_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
                        Your dataset root directory
  --model MODEL         which model
  --lr LR               learning rate
  --gpu GPU [GPU ...]   gpu device
  --epochs EPOCHS       num of epoch
  --num-classes NUM_CLASSES
                        The number of classes for your classification problem
  --train-batch-size TRAIN_BATCH_SIZE
                        The batch size for training data
  --dev-batch-size DEV_BATCH_SIZE
                        The batch size for validation data
  --num-workers NUM_WORKERS
                        The number of worker while training
  --logs LOGS           Directory to save all your checkpoint.pth
  --img-size IMG_SIZE   Input image size
```

Recommended training command:

```
python train.py --data-root <path_to_data> --gpu 0 1 --logs ./efficientnet_b4 --lr 0.005 --model efficientnet-b4 --img-size 380 --train-batch-size 21 --dev-batch-size 8
```

The logging directory will be generated in the path you specified for `--logs`. Inside this logging directory you can find:
* `checkpoints/`: All the training checkpoints will be saved inside here. Checkpoints is saved every 10 epochs and `best_model.pth` save the current best model based on evaluation accuracy.
* `events/`: Tensorboard event files, you can visualize your experiment by `tensorboard --logdir events/`
* `logger/`: Some information that print on screen while training will be log into here for your future reference.

## Testing
You can test your training results by running this command:
```
python test.py [-h] [--data-root DATA_ROOT] [--ckpt CKPT] [--img-size IMG_SIZE]
               [--num-classes NUM_CLASSES] [--net NET] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
                        data root dir
  --ckpt CKPT           checkpoint path
  --img-size IMG_SIZE   image size in model
  --num-classes NUM_CLASSES
                        number of classes
  --net NET             which model
  --gpu GPU             gpu id
```

## Submit the results
Run this command to `zip` your submission file:
```
zip answer.zip answer.txt
```
You can upload `answer.zip` to the challenge. Then you can get your testing score.

## Pre-trained models

Click into [Releases](https://github.com/joycenerd/bird-images-classification/releases). Under **EfficientNet-b4 model** download `efficientnet-b4_best_model.pth`. This pre-trained model get accuracy 72.53% on the test set.

Recommended testing command:
```
python test.py --data-root <path_to_data> --ckpt <path_to_checkpoint> --img-size 380 --net efficientnet-b4 --gpu 0
```

`answer.txt` will be generated in this directory. This file is the submission file.

## Inference
To reproduce our results, run this command:
```
python inference.py --data-root <path_to_data> --ckpt <pre-trained_model_path> --img-size 380 --net efficientnet-b4 --gpu 0
```

## Reproducing Submission

To reproduce our submission without retraining, do the following steps

1. [Getting the code](#getting-the-code)
2. [Install the dependencies](#requirements)
2. [Download the data](#dataset)
4. [Download pre-trained models](#pre-trained-models)
3. [Inference](#inference)
4. [Submit the results](#submit-the-results)

## Results

Our model achieves the following performance:

|     | EfficientNet-b4 w/o sched | EfficientNet-b4 with sched |
|-----|---------------------------|----------------------------|
| acc | 55.29%                    | 72.53%                     |

## Citation
If you find our work useful in your project, please cite:

```bibtex
@misc{
    title = {bird_image_classification},
    author = {Zhi-Yi Chin},
    url = {https://github.com/joycenerd/bird-images-classification},
    year = {2021}
}
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
