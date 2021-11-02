# SI3DP

SI3DP: Source Identification Challenges and Benchmark for Consumer-Level 3D Printer Forensics


## SOFTWARE 


Python 3.6.9 ~ 3.7.9

CUDA Version 10.2.89

cuddn 7.6.5

(See details of python package in `requirements.txt`)

1. Nvidia driver, CUDA toolkit 10.2, install Anaconda

2. Install pytorch 

        conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

3. Install various necessary packages

        pip install opencv-contrib-python kaggle resnest geffnet albumentations pillow scikit-learn scikit-image pandas tqdm pretrainedmodels
        
4. Install git and LR related packages

        conda install git
        pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

## DATA SETUP 

Based on the `./data` sub-path

1. Printer, Filament, Quality task 

        `./data`



## Training


It was written based on the SIIM-ISIC Melanoma Classification kernel structure. https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412



When using Terminal, directly execute the code below after setting the path

        python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

When using pycharm:

        Menu Run 
        -> Edit Configuration 
        -> Check train.py in Script path
        -> Go to parameters and enter the following

        --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

        -> Running/debugging after check Apply button

As training progresses, the best and final weights are saved in the folder `./weights/`. Learning logs are saved in the `./logs/` folder.


## (Optional) Evaluating

The learned model is subjected to k-fold cross validation. You can use the model used for training earlier, or you can evaluate it by specifying the model in `--model-dir`.

The evaluation results are saved in `./logs/`, and the combined results (out-of-folds) are saved in the `./oofs/` folder.

