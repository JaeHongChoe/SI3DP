import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from utils.util import *
# from torchsampler import ImbalancedDatasetSampler
from dataset import get_df_3dprint, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC,Effnet_MMC_Multitask,Effnet_MMC_Multi_Modal,Effnet_MMC_Multi_Modal_Single_Task

Precautions_msg = '(Precautions for use) ---- \n'


'''
- train.py

Code containing the entire process

#### Manual ####
If you are using Terminal,set the path and run the code below directlypycharm

In the case of pycharm:  
Verify that [Run -> Edit Configuration -> train.py] is selected
-> Go to parameters and enter below -> Run/debug after click apply
ex)Printer task 
--kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close

*** def parse_args(): There is all the information about the execution parameters.  
*** def run(): A function that contains the whole process of learning. You can improve performance by applying various tricks here.
** def main(): Run after distributing the data divided by the fold to the [def run].
* def train_epoch(), def val_epoch() : Correction after understanding completely

 MMCLab, 허종욱, 2020python 



### 3D Project Terminal ###

<Closed up Setting>
- Printer task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close
- Filament task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 2 --img-type close
- Quality task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 3 --img-type close
- Device task 
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 4 --img-type close
- Reprint task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 5 --img-type close

<Fullshot Setting>
- Printer task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type full
- Filament task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 2 --img-type full
- Quality task 
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 3 --img-type full
- Device task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 4 --img-type full 
- Reprint task
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 5 --img-type full

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 1 --batch-size 32 --n-epochs 50
- Multi-Task(Device & Quality)
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 3 --batch-size 32 --n-epochs 50

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --batch-size 32 --n-epochs 50
- Multi-Modal-Task(Device & Printer)
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 3 --batch-size 32 --epoch 50
- Multi-Modal-Task(Device & Quality)
python train.py --kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 1 --batch-size 32 --epoch 50
'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : Distinguished name with overall information about experimental settings


    parser.add_argument('--data-dir', type=str, default='./data/')
    # base data folder ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # Data Detail Folder ex) 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='300')
    # Image data size to input

    parser.add_argument('--enet-type', type=str, required=True)
    # Network name to apply to learning
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    # Medium layer size when using meta data

    parser.add_argument('--out-dim', type=int, default=5)
    # Model output dimension

    parser.add_argument('--out-dim2', type=int, default=5)
    # Model output2 dimension

    parser.add_argument('--task-type', type=int, default=1, required=True)
    # Setting the task type to experiment with

    parser.add_argument('--side-task-type', type=int, default=0)
    # Setting the multi-task type to experiment with

    parser.add_argument('--DEBUG', action='store_true')
    # Parameters for Debugging (Hold the experimental epoch at 5)

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # GPU number to use for learning

    parser.add_argument('--k-fold', type=int, default=4)
    # data cross-validation
    # Specifies the k value of the k-fold

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/


    parser.add_argument('--accumulation_step', type=int, default=1)
    # Gradient accumulation step
    # When GPU memory is low, the batch is split, processed, and then merged
    # If batch is 30, model updates are combined to 60

    parser.add_argument('--model-dir', type=str, default='./weights')
    # Specify weight storage folder
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # Whether to use external data in addition to the original data

    parser.add_argument('--img-type', type=str, required=True)
    # Whether to use {close, full, both} image data

    parser.add_argument('--batch-size', type=int, default=16)
    # batch size

    parser.add_argument('--num-workers', type=int, default=4)
    # Number of threads from which data is read

    parser.add_argument('--init-lr', type=float, default=3e-5)
    # Initial learning rate

    parser.add_argument('--n-epochs', type=int, default=30)
    # number of epochs

    args, _ = parser.parse_known_args()
    return args




def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)

    # multi-task multi-modal
    if args.img_type == 'both' and args.side_task_type:
        for i, (close_image, full_image, target_d, target_q) in enumerate(bar):
            optimizer.zero_grad()
            close_image, full_image, target_d, target_q = close_image.to(device), full_image.to(device), target_d.to(
                device), target_q.to(device)

            logits1, logits2, logits3 = model(close_image, full_image)

            loss1_cf_d = criterion(logits1, target_d)
            loss2_cf_q = criterion(logits2, target_q)
            loss3_cf_q = criterion(logits3, target_q)

            alpha = 0.8
            beta = 0.2
            loss = (loss1_cf_d + loss3_cf_q) * alpha + beta * (loss2_cf_q)
            loss.backward()


            # gradient accumulation (When memory is low)
            if args.accumulation_step:
                if (i + 1) % args.accumulation_step == 0:
                    optimizer.step()
                    # optimizer.zero_grad()
            else:
                optimizer.step()
                # optimizer.zero_grad()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    # single-task multi-modal
    elif args.img_type == 'both' and args.side_task_type == False:
        for i, (close_image, full_image, target_d) in enumerate(bar):
            optimizer.zero_grad()
            close_image, full_image, target_d = close_image.to(device), full_image.to(device), target_d.to(
                device)


            logits, logits2 = model(close_image, full_image)

            loss = (criterion(logits, target_d) + criterion(logits2, target_d))/2
            loss.backward()

            # gradient accumulation (When memory is low)
            if args.accumulation_step:
                if (i + 1) % args.accumulation_step == 0:
                    optimizer.step()
                    # optimizer.zero_grad()
            else:
                optimizer.step()
                # optimizer.zero_grad()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    # multi-task single-modal
    elif args.side_task_type:
        for i, (data, target1, target2) in enumerate(bar):

            optimizer.zero_grad()
            data, target1, target2 = data.to(device), target1.to(device), target2.to(device)
            logits1, logits2 = model(data)

            loss1 = criterion(logits1, target1)
            loss2 = criterion(logits2, target2)

            loss = (loss1 + loss2) / 2
            loss.backward()


            # gradient accumulation (When memory is low)
            if args.accumulation_step:
                if (i + 1) % args.accumulation_step == 0:
                    optimizer.step()
                    # optimizer.zero_grad()
            else:
                optimizer.step()
                # optimizer.zero_grad()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    # single-task single-modal
    else:
        for i, (data, target) in enumerate(bar):
            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)

            loss.backward()

            # gradient accumulation (When memory is low)
            if args.accumulation_step:
                if (i + 1) % args.accumulation_step == 0:
                    optimizer.step()
                    #optimizer.zero_grad()
            else:
                optimizer.step()
                #optimizer.zero_grad()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))



    train_loss = np.mean(train_loss)
    return train_loss


def val_epoch(model, loader, n_test=1):
    '''

    Output:
    val_loss, acc, TARGETS, PROBS
    '''

    model.eval()
    val_loss = []
    PROBS = []
    TARGETS = []
    PRINTERS = []
    OBJ_IDX = []
    LOGITS2 = []
    PROBS2 = []
    TARGETS2 = []


    with torch.no_grad():
        # multi-task multi-modal
        if args.img_type == 'both' and args.side_task_type:
            for (close_image, full_image, target1, target2, obj_id,printer_id) in tqdm(loader):
                close_image, full_image, target1, target2 = close_image.to(device), full_image.to(
                    device), target1.to(
                    device), target2.to(device)

                logits = torch.zeros((close_image.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((close_image.shape[0], args.out_dim)).to(device)

                for I in range(n_test):
                    l, l_2, l_3 = model(get_trans(close_image, I), get_trans(full_image, I))
                    logits += l
                    probs += l.softmax(1)

                logits /= n_test
                probs /= n_test

                loss = criterion(l, target1)

                PROBS.append(probs.detach().cpu())
                TARGETS.append(target1.detach().cpu())
                OBJ_IDX.append(obj_id.detach().cpu())
                val_loss.append(loss.detach().cpu().numpy())

        # single-task multi-modal
        elif args.img_type == 'both' and args.side_task_type == False:
            for (close_image, full_image, target, obj_id,printer_id) in tqdm(loader):
                close_image, full_image, target = close_image.to(device), full_image.to(device), target.to(device)

                logits = torch.zeros((close_image.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((close_image.shape[0], args.out_dim)).to(device)

                for I in range(n_test):
                    l, l2 = model(get_trans(close_image, I), get_trans(full_image, I))
                    logits += l
                    probs += l.softmax(1)

                logits /= n_test
                probs /= n_test

                loss = (criterion(l, target) + criterion(l2, target))/2

                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())
                OBJ_IDX.append(obj_id.detach().cpu())
                val_loss.append(loss.detach().cpu().numpy())


        # multi-task single-modal
        elif args.side_task_type:
            for (data, target, target2, obj_idx,printer_id) in tqdm(loader):
                data, target, target2, obj_idx,printer_id = data.to(device), target.to(device), target2.to(
                    device), obj_idx.to(device),printer_id.to(device)

                logits1 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                logits2 = torch.zeros((data.shape[0], args.out_dim2)).to(device)

                probs1 = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs2 = torch.zeros((data.shape[0], args.out_dim2)).to(device)

                for I in range(n_test):
                    l, l_2 = model(get_trans(data, I))
                    logits1 += l
                    logits2 += l_2

                    probs1 += l.softmax(1)
                    probs2 += l_2.softmax(1)


                logits1 /= n_test
                logits2 /= n_test


                probs1 /= n_test
                probs2 /= n_test


                PROBS.append(probs1.detach().cpu())
                TARGETS.append(target.detach().cpu())

                LOGITS2.append(logits2.detach().cpu())
                PROBS2.append(probs2.detach().cpu())
                TARGETS2.append(target2.detach().cpu())

                PRINTERS.append(printer_id.detach().cpu())
                OBJ_IDX.append(obj_idx.detach().cpu())

                loss1 = criterion(logits1, target)
                loss2 = criterion(logits2, target2)

                loss = (loss1 + loss2) / 2

                val_loss.append(loss.detach().cpu().numpy())

        # single-task single-modal
        else:
            for (data, target,obj_idx,printer_id) in tqdm(loader):

                data, target,obj_idx,printer_id = data.to(device), target.to(device),obj_idx.to(device),printer_id.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)

                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l

                    probs += l.softmax(1)

                logits /= n_test
                probs /= n_test

                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())
                PRINTERS.append(printer_id.detach().cpu())
                OBJ_IDX.append(obj_idx.detach().cpu())

                loss = criterion(logits, target)
                val_loss.append(loss.detach().cpu().numpy())



    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    OBJ_IDX = torch.cat(OBJ_IDX).numpy()

    unique_idx = np.unique(OBJ_IDX).astype(np.int64)

    for u_id in unique_idx.tolist():
        res_list = np.where(OBJ_IDX == u_id)
        mean_prob = PROBS[res_list].mean(axis = 0)
        PROBS[res_list] = mean_prob


    # accuracy
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.

    return val_loss, acc,TARGETS, PROBS.argmax(1)

def run(fold, df, transforms_train, transforms_val, df_full = None):
    '''
    Learning progress main function

    :param fold: The partition number to be used for value in cross-validation
    :param df: Full Data List for DataFrame Learning
    :param meta_features, n_meta_features: Whether to use additional information other than images
    :param transforms_train, transforms_val: Dataset transform function
    '''

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)

    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        #
        if len(df_train) % args.batch_size == 1:
            df_train = df_train.sample(len(df_train)-1)
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)


    # Read Dataset
    if args.img_type == 'both':
        dataset_train = MMC_ClassificationDataset(df_train, 'train', csv2 = df_full, transform=transforms_train,side_task = args.side_task_type,img_type = args.img_type)
        dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', csv2 = df_full, transform=transforms_val,task_type = args.task_type,side_task = args.side_task_type,img_type = args.img_type)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    else:
        dataset_train = MMC_ClassificationDataset(df_train, 'train', transform=transforms_train,side_task = args.side_task_type,img_type = args.img_type)
        dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', transform=transforms_val,task_type = args.task_type,side_task = args.side_task_type,img_type = args.img_type)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    acc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')
    
    #
    # If there is a pretrained file
    if os.path.isfile(model_file2):
        if args.side_task_type:
            model = ModelClass(
                args.enet_type,
                out_dim=args.out_dim,
                out_dim2=args.out_dim2,
                pretrained=True
            )
            model.load_state_dict(torch.load(model_file)) # Model Pathing
        else:
            model = ModelClass(
                args.enet_type,
                n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
                out_dim=args.out_dim,
                pretrained=True
            )
            model.load_state_dict(torch.load(model_file)) # Model Pathing
    else:
        if args.side_task_type:
            model = ModelClass(
                args.enet_type,
                out_dim=args.out_dim,
                out_dim2=args.out_dim2,
                pretrained=True
            )
        else:
            model = ModelClass(
                args.enet_type,
                n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
                out_dim=args.out_dim,
                pretrained=True
            )

    model = model.to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    if DP:
        model = nn.DataParallel(model)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer)

        val_loss, acc,targets,probs = val_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, Acc: {(acc):.4f}'

        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch == 2:
            scheduler_warmup.step() # bug workaround
            
        if acc > acc_max:
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), model_file)
            acc_max = acc

    torch.save(model.state_dict(), model_file2)


def main():

    if args.task_type == 1:
        args.out_dim = 5
    elif args.task_type == 2:
        args.out_dim = 7
    elif args.task_type == 3:
        args.out_dim = 3
    elif args.task_type == 4:
        args.out_dim = 14
    elif args.task_type == 5:
        args.out_dim = 2
    else:
        args.out_dim = 0

    if args.side_task_type == 1:
        args.out_dim2 = 5
    elif args.side_task_type == 2:
        args.out_dim2 = 7
    elif args.side_task_type == 3:
        args.out_dim2 = 3
    elif args.side_task_type == 4:
        args.out_dim2 = 14
    elif args.side_task_type == 5:
        args.out_dim2 = 2
    else:
        args.out_dim2 = 0

    '''
    ####################################################
    # 3d printer dataset : dataset.get_df_3d print
    ####################################################
    '''
    if args.img_type == 'both':
        df_train_close, df_train_full, df_test = get_df_3dprint(
            k_fold = args.k_fold,
            out_dim = args.out_dim,
            out_dim2 = args.out_dim2,
            data_dir = args.data_dir,
            data_folder = args.data_folder,
            task_type = args.task_type,
            img_type = args.img_type,
            side_task = args.side_task_type
        )
    else:
        df_train, df_test = get_df_3dprint(
            k_fold = args.k_fold,
            out_dim = args.out_dim,
            out_dim2 = args.out_dim2,
            data_dir = args.data_dir,
            data_folder = args.data_folder,
            task_type = args.task_type,
            img_type = args.img_type,
            side_task = args.side_task_type
        )

    # Recall model transforms
    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = range(args.k_fold)

    if args.img_type == 'both':
        for fold in folds:
            run(fold, df_train_close , transforms_train, transforms_val, df_full = df_train_full)
    else:
        for fold in folds:
            run(fold, df_train, transforms_train, transforms_val)


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # make argument
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # Recall Network Type Settings
    if args.enet_type == 'resnest101':
        ModelClass = Resnest_MMC
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_MMC
    elif 'efficientnet' in args.enet_type:
        if args.img_type == 'both' and args.side_task_type:
            ModelClass = Effnet_MMC_Multi_Modal
        elif args.img_type == 'both' and args.side_task_type == False:
            ModelClass = Effnet_MMC_Multi_Modal_Single_Task
        elif args.side_task_type:
            ModelClass = Effnet_MMC_Multitask
        else:
            ModelClass = Effnet_MMC
    else:
        raise NotImplementedError()

    # Whether to use a multi-GPU
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # Random seed settings for experimental reproduction
    set_seed(2359)
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    # perform the main function
    main()