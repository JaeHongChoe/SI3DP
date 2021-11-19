# SI3DP

SI3DP: Source Identification Challenges and Benchmark for Consumer-Level 3D Printer Forensics


## SOFTWARE 


Python 3.7.0 ~ 3.7.9

CUDA Version 11.0

(See details of python package in `requirements.txt`)

1. Nvidia driver, CUDA toolkit 11.0, install Anaconda

2. Install pytorch 

        conda install pytorch torchvision cudatoolkit=11.0 -c pytorch

3. Install various necessary packages

        pip install opencv-contrib-python kaggle resnest geffnet albumentations pillow scikit-learn scikit-image pandas tqdm pretrainedmodels
        
4. Install git and LR related packages

        conda install git
        pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

## DATA SETUP 

Click [here](https://forms.gle/jCEDjdL5caaU1QrRA) to submit your information to download Dataset and csv.

Download [7zip](https://www.7-zip.org) and unzip your dataset. Create a `data` folder with `SI3DP` folder as a sub-path.

Please move the full shot and close-up dataset folders in the SI3DP folder.

Based on the `./data` sub-path `SI3DP/{train_close,train_full,train_close.csv,train_full.csv}`

------

1. Printer, Filament, Quality task 

        ./data/SI3DP/train_(close / full).csv

2. Device task
        
        ./data/SI3DP/train_(close / full)_device.csv

3. Reprint task

        ./data/SI3DP/train_(close / full)_reprint.csv

## Training

When using Terminal, directly execute the code below after setting the path

        python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

When using pycharm:

        Menu Run 
        -> Edit Configuration 
        -> Check train.py in Script path
        -> Go to parameters and enter the following

        --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

        -> Running/debugging after check Apply button

As training progresses, the best and final weights are saved in the folder `./weights/`. Learning logs are saved in the `./logs/` folder.

The order of side task type and task type is 1 : Printer, 2 : Filament, 3 : Quality, 4 : Device, 5 : Reprint.

```
<Closed up Setting>
- Printer task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close
- Filament task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 2 --img-type close
- Quality task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 3 --img-type close
- Device task 
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 4 --img-type close
- Reprint task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 5 --img-type close

<Fullshot Setting>
- Printer task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type full
- Filament task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 2 --img-type full
- Quality task 
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 3 --img-type full
- Device task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 4 --img-type full 
- Reprint task
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 5 --img-type full

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 1 --batch-size 32 --n-epochs 50
- Multi-Task(Device & Quality)
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 3 --batch-size 32 --n-epochs 50

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --batch-size 32 --n-epochs 50
- Multi-Modal-Task(Device & Printer)
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 3 --batch-size 32 --epoch 50
- Multi-Modal-Task(Device & Quality)
python train.py --kernel-type test --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 1 --batch-size 32 --epoch 50
```

## Evaluating

The learned model is subjected to k-fold cross validation. You can use the model used for training earlier, or you can evaluate it by specifying the model in `--model-dir`.

The evaluation results are saved in `./logs/`, and the combined results (out-of-folds) are saved in the `./oofs/` folder.

```
<Closed up Setting>
- Printer task
python evaluate.py --kernel-type printer_close --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 1 --img-type close
- Filament task
python evaluate.py --kernel-type filament_close --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 2 --img-type close
- Quality task
python evaluate.py --kernel-type quality_close --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 3 --img-type close
- Device task
python evaluate.py --kernel-type device_close --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close
- Reprint task
python evaluate.py --kernel-type reprint_close --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 5 --img-type close

<Fullshot Setting>
- Printer task
python evaluate.py --kernel-type printer_full --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 1 --img-type full
- Filament task
python evaluate.py --kernel-type filament_full --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 2 --img-type full
- Quality task
python evaluate.py --kernel-type quality_full --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 3 --img-type full
- Device task
python evaluate.py --kernel-type device_full --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type full
- Reprint task
python evaluate.py --kernel-type reprint_full --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 5 --img-type full

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python evaluate.py --kernel-type multi_d_p --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 1 --batch-size 32 --n-epochs 50
- Multi-Task(Device & Quality)
python evaluate.py --kernel-type multi_d_q --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 3 --batch-size 32 --n-epochs 50

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python evaluate.py --kernel-type modal_d --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --batch-size 32 --n-epochs 50
- Multi-Modal-Task(Device & Printer)
python evaluate.py --kernel-type modal_d_p --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 1 --batch-size 32 --epoch 50
- Multi-Modal-Task(Device & Quality)
python evaluate.py --kernel-type modal_d_q --data-folder SI3DP/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 3 --batch-size 32 --epoch 50
```




## Acknowledgement 

Overall code structure is borrowed from [this code](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412). If you want to cite our Datasets [paper](https://dl.acm.org/doi/10.1145/3474085.3475316) and code, you can use these:

```bibtex
@inproceedings{shim2021si3dp,
  title={SI3DP: Source Identification Challenges and Benchmark for Consumer-Level 3D Printer Forensics},
  author={Shim, Bo Seok and Shin, Yoo Seung and Park, Seong Wook and Hou, Jong-Uk},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1721--1729},
  year={2021}
}
```
