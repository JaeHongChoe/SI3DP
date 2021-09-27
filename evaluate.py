import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import get_df_3dprint, get_transforms, MMC_ClassificationDataset
from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC,Effnet_MMC_Multitask,Effnet_MMC_Multi_Modal,Effnet_MMC_Multi_Modal_Single_Task
from utils.util import *
from utils.torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

Precautions_msg = ' '


'''
- evaluate.py

Code that evaluates the trained model
We use the validation set we looked at during training, not the test set.

#### Manual ####
If you are using Terminal,set the path and run the code below directlypycharm


In the case of pycharm:  
Verify that [Run -> Edit Configuration -> evaluate.py] is selected
-> Go to parameters and enter below -> Run/debug after click apply
ex)Printer task 
--kernel-type test --data-folder sampled_face/ --enet-type tf_efficientnet_b3_ns --n-epochs 50 --batch-size 32 --task-type 1 --img-type close


#### 3D Project Terminal version #### 
<Closed up Setting>
- Printer task
python evaluate.py --kernel-type printer_close --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 1 --img-type close
- Filament task
python evaluate.py --kernel-type filament_close --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 2 --img-type close
- Quality task
python evaluate.py --kernel-type quality_close --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 3 --img-type close
- Device task
python evaluate.py --kernel-type device_close --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close
- Reprint task
python evaluate.py --kernel-type reprint_close --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 5 --img-type close

<Fullshot Setting>
- Printer task
python evaluate.py --kernel-type printer_full --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 1 --img-type full
- Filament task
python evaluate.py --kernel-type filament_full --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 2 --img-type full
- Quality task
python evaluate.py --kernel-type quality_full --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 3 --img-type full
- Device task
python evaluate.py --kernel-type device_full --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type full
- Reprint task
python evaluate.py --kernel-type reprint_full --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 5 --img-type full

<Multi-Task Setting>
- Multi-Task(Device & Printer)
python evaluate.py --kernel-type multi_d_p --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 1 --batch-size 32 --n-epochs 50
- Multi-Task(Device & Quality)
python evaluate.py --kernel-type multi_d_q --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type close --side-task-type 3 --batch-size 32 --n-epochs 50

<(Multi or Single) Modal-Task  Setting>
- Single-Modal-Task(Device)
python evaluate.py --kernel-type modal_d --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --batch-size 32 --n-epochs 50
- Multi-Modal-Task(Device & Printer)
python evaluate.py --kernel-type modal_d_p --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 1 --batch-size 32 --epoch 50
- Multi-Modal-Task(Device & Quality)
python evaluate.py --kernel-type modal_d_q --data-folder sampled_face/ --enet-type tf_efficientnet_b3 --task-type 4 --img-type both --side-task-type 3 --batch-size 32 --epoch 50
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=300)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dim', type=int,default=5)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--oof-dir', type=str, default='./oofs')
    parser.add_argument('--side-task-type', type=int, default=0)

    parser.add_argument('--k-fold', type=int, default=4)
    parser.add_argument('--eval', type=str, choices=['best', 'best_no_ext', 'final'], default="final")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    parser.add_argument('--task-type', type=int, default=1, required=True) # Task number to verify
    # 1: Printer, 2: Device, 3: Filament 4: Quality 5: Reprint

    parser.add_argument('--img-type', type=str, required=True) # 검증 데이터 타입
    # img_type = 'close', 'full', 'both'


    args, _ = parser.parse_known_args()
    return args

def visualization(confusion_matrix,obj = False):
    device_label = ['2F1', '2F2', '2F3', '2F4', '3C1', '3C2', '3C3', '3C4', 'F1', 'F2', 'F3', 'F4', 'MX', 'RP']
    printer_label = ['210F', '320C', 'Finder', 'MethodX', 'Replicator']
    filament_label = ['ABS', 'ASA', 'NYLON', 'PETG', 'PLA', 'PLA+', 'ToughPLA']
    quality_label = ['Layer','Shell']
    obj_label = ['Bullet','Ironman','Key','Teeth']
    reprint_label = ['True', 'False']

    if args.task_type == 1:
        val_label = printer_label
        title = 'Printer Model'
    elif args.task_type == 2:
        val_label = filament_label
        title = 'Filament Model'
    elif args.task_type == 3:
        val_label = quality_label
        title = 'Quality Model'
    elif args.task_type == 4:
        val_label = device_label
        title = 'Device Model'
    elif args.task_type == 5:
        val_label = reprint_label
        title = 'Reprint Model'

    mpl.style.use('seaborn')


    if obj:
        confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=val_label, columns=obj_label)
    elif args.task_type == 3:
        confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=val_label, columns=printer_label)
    else:
        total = np.sum(confusion_matrix.numpy(), axis=1)
        confusion_matrix = confusion_matrix / total[:, None]
        confusion_matrix = pd.DataFrame(confusion_matrix.numpy(), index=val_label, columns=val_label)


    fig = plt.figure(figsize=(12, 9))
    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size":10} ,fmt='.2f', cmap=cmap)

    if obj:
        plt.xticks(np.arange(len(obj_label)) + 0.5, obj_label, size=30, rotation='vertical')
    elif args.task_type == 3:
        plt.xticks(np.arange(len(printer_label)) + 0.5, printer_label, size=30, rotation='vertical')
    else:
        plt.xticks(np.arange(len(val_label)) + 0.5, val_label, size=25, rotation='vertical')
    plt.yticks(np.arange(len(val_label)) + 0.5, val_label, size=25, rotation='horizontal')
    plt.title(title, fontsize=30)
    plt.savefig('confusion_matrix_device_close_real.png', dpi=100, bbox_inches='tight')

    confusion_matrix.to_csv(os.path.join('./con_matrix',  f'conf_{args.kernel_type}_best_fold{fold}_accuracy{Accuracy}.csv'))
    plt.show()
    plt.close()

def Task_name():

    if args.img_type == 'close':
        image_name = 'Closeshot'
    elif args.img_type == 'full':
        image_name = 'Fullshot'
    else:
        image_name = 'Modal(Close & Full)'

    if args.task_type == 1:
        task_name = 'Main-task : Printer'
    elif args.task_type == 2:
        task_name = 'Main-task : Filament'
    elif args.task_type == 3:
        task_name = 'Main-task : Quality'
    elif args.task_type == 4:
        task_name = 'Main-task : Device'
    elif args.task_type == 5:
        task_name = 'Main-task : Reprint'

    if args.side_task_type == 1:
        side_task_name = 'Side-task : Printer'
    elif args.side_task_type == 2:
        side_task_name = 'Side-task : Filament'
    elif args.side_task_type == 3:
        side_task_name = 'Side-task : Quality'
    elif args.side_task_type == 4:
        side_task_name = 'Side-task : Device'
    elif args.side_task_type == 5:
        side_task_name = 'Side-task : Reprint'
    else:
        side_task_name = ''

    name = image_name + ' / ' + task_name + ' / ' + side_task_name

    return name

def val_epoch(model, loader, n_test=1):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    OBJ_IDX = []
    EXTRA = []
    ConT = []
    ConP =[]
    ConEXTRA = []

    with torch.no_grad():

        if args.img_type == 'both' and args.side_task_type:
            for (close_image, full_image, target1, target2, obj_id,extra_id) in tqdm(loader):
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

                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target1.detach().cpu())
                EXTRA.append(extra_id.detach().cpu())
                OBJ_IDX.append(obj_id.detach().cpu())
                val_loss.append(loss.detach().cpu().numpy())
        elif args.img_type == 'both' and args.side_task_type == False:
            for (close_image, full_image, target1, obj_id,extra_id) in tqdm(loader):
                close_image, full_image, target1 = close_image.to(device), full_image.to(device), target1.to(device)

                logits = torch.zeros((close_image.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((close_image.shape[0], args.out_dim)).to(device)

                for I in range(n_test):
                    l, l2 = model(get_trans(close_image, I), get_trans(full_image, I))
                    logits += l
                    probs += l.softmax(1)

                logits /= n_test
                probs /= n_test

                loss = criterion(l, target1)

                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target1.detach().cpu())
                OBJ_IDX.append(obj_id.detach().cpu())
                val_loss.append(loss.detach().cpu().numpy())
                EXTRA.append(extra_id.detach().cpu())

        elif args.side_task_type:
            for (data, target, target2, obj_idx,extra_id) in tqdm(loader):
                data, target, target2, obj_idx,extra_id = data.to(device), target.to(device), target2.to(
                    device), obj_idx.to(device),extra_id.to(device)

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
                EXTRA.append(extra_id.detach().cpu())
                OBJ_IDX.append(obj_idx.detach().cpu())

                loss1 = criterion(logits1, target)
                loss2 = criterion(logits2, target2)

                loss = (loss1 + loss2) / 2

                val_loss.append(loss.detach().cpu().numpy())
        else:
            for (data, target,obj_idx,extra_id) in tqdm(loader):

                data, target,obj_idx,extra_id = data.to(device), target.to(device),obj_idx.to(device),extra_id.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)

                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l

                    probs += l.softmax(1)

                logits /= n_test
                probs /= n_test

                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())
                EXTRA.append(extra_id.detach().cpu())
                OBJ_IDX.append(obj_idx.detach().cpu())

                loss = criterion(logits, target)
                val_loss.append(loss.detach().cpu().numpy())

    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    EXTRA = torch.cat(EXTRA).numpy()
    OBJ_IDX = torch.cat(OBJ_IDX).numpy()

    unique_idx = np.unique(OBJ_IDX).astype(np.int64)

    for u_id in unique_idx.tolist():
        res_list = np.where(OBJ_IDX == u_id)
        ConT.append(np.unique(TARGETS[res_list]))
        ConEXTRA.append(np.unique(EXTRA[res_list]))
        mean_prob = PROBS[res_list].mean(axis = 0)
        ConP.append(mean_prob.argmax())
        PROBS[res_list] = mean_prob


    if args.task_type == 3 or args.task_type == 1:
        ConT = np.concatenate(ConT)
        ConEXTRA = np.concatenate(ConEXTRA)

        if args.task_type == 1:
            ConT = np.array((ConT, ConEXTRA)).transpose(1, 0)

            #return LOGITS, PROBS.argmax(1), TARGETS_EXTRA_IDX, ConP, ConT
            return ConP, ConT
        LAYER_T = np.where(ConT == 2, 0,ConT)
        LAYER_P = np.where(np.array(ConP) == 2, 0, np.array(ConP))

        SHELL_T = np.where(ConT == 2, 1, ConT)
        SHELL_P = np.where(np.array(ConP) == 2, 1, np.array(ConP))

        ConT = np.array((ConT, ConEXTRA)).transpose(1, 0)
        LAYER_T = np.array((LAYER_T, ConEXTRA)).transpose(1, 0)
        SHELL_T = np.array((SHELL_T, ConEXTRA)).transpose(1, 0)

        return ConP, ConT,LAYER_P,LAYER_T,SHELL_P,SHELL_T
    else:

        return ConP, ConT


def main():
    '''
        args.task_type
        1: printer
        2: filament
        3: quality
        4: device-level
        5: reprint detection

    :return:
    '''

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
        df_train, df_train_full, df_test = get_df_3dprint(
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

    transforms_train, transforms_val = get_transforms(args.image_size)

    PROBS = []
    TARGETS = []
    dfs = []

    Layer_answer = []
    Shell_answer = []

    # create confusion matrix
    nb_classes = args.out_dim
    if args.task_type == 3:
        confusion_matrix = torch.zeros(2, 5)
    else:
        confusion_matrix = torch.zeros(nb_classes, nb_classes)

    folds = range(args.k_fold)

    for fold in folds:
        print(f'Evaluate data fold{str(fold)}')
        df_valid = df_train[df_train['fold'] == fold]

        # In batch_normalization, an error can occur if batch size 1, so discard one data
        if len(df_valid) % args.batch_size == 1:
            df_valid = df_valid.sample(len(df_valid)-1)

        if args.DEBUG:
            df_valid = df_valid[df_valid['fold'] == fold].sample(args.batch_size * 5)


        # Read Dataset
        if args.img_type == 'both':
            dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', csv2=df_train_full,
                                                      transform=transforms_val, task_type=args.task_type,
                                                      side_task=args.side_task_type, img_type=args.img_type)

            valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        else:
            dataset_valid = MMC_ClassificationDataset(df_valid, 'valid', transform=transforms_val,
                                                      task_type=args.task_type, side_task=args.side_task_type,
                                                      img_type=args.img_type)
            valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,
                                                       num_workers=args.num_workers)

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        elif args.eval == 'best_no_ext':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_no_ext_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')


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


        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        '''
        ####################################################
        # evaluation function for data : val_epoch
        ####################################################
        '''
        if args.task_type == 3:
            this_PROBS, this_TARGETS, this_LayerP, this_LayerT, this_ShellP, this_ShellT = val_epoch(model, valid_loader, n_test=1)
            copyT = this_TARGETS.copy()
            this_LayerT[:, 0] = np.where(this_LayerP == this_LayerT[:, 0], 1, 0)
            this_ShellT[:, 0] = np.where(this_ShellP == this_ShellT[:, 0], 1, 0)
            copyT[:, 0] = np.where(this_PROBS == this_TARGETS[:, 0], 1, 0)

            Layer_answer.append(this_LayerT)
            Shell_answer.append(this_ShellT)

            PROBS.append(this_PROBS)
            TARGETS.append(this_TARGETS)

        else:
            this_PROBS, this_TARGETS = val_epoch(model, valid_loader, n_test=1)
            PROBS.append(this_PROBS)
            TARGETS.append(this_TARGETS)
            dfs.append(df_valid)

            if args.task_type == 1:
                for t, p in zip(this_TARGETS[:,0], this_PROBS):
                    confusion_matrix[t, p] += 1

            else:
                for t, p in zip(this_TARGETS, this_PROBS):
                    confusion_matrix[t, p] += 1



    if args.task_type == 3:
        Shell_answer = np.concatenate(Shell_answer)
        Layer_answer = np.concatenate(Layer_answer)

        for u_id in [0, 1, 2, 3, 4]:
            res_layer = np.where(Layer_answer[:, 1] == u_id)
            res_shell = np.where(Shell_answer[:, 1] == u_id)

            save_layer = np.zeros((np.array(res_layer).shape[1]))
            save_shell = np.zeros((np.array(res_shell).shape[1]))
            for i,j in zip(res_layer,res_shell):
                save_layer += Layer_answer[i, 0]
                save_shell += Shell_answer[j, 0]


            mean_layer = save_layer.mean()
            mean_shell = save_shell.mean()

            confusion_matrix[0, u_id] = mean_layer
            confusion_matrix[1, u_id] = mean_shell

        PROBS = np.concatenate(PROBS)
        TARGETS = np.concatenate(TARGETS)

        Accuracy = (PROBS == TARGETS[:,0]).mean() * 100.

        content = time.ctime() + ' ' + f'Eval {args.eval}:\nAccuracy : {Accuracy:.5f}\n'

        micro_averaged_precision = metrics.precision_score(TARGETS[:,0], PROBS, average='micro')

        print(f"\nTask name : {Task_name()}")
        print(f"Micro-Averaged Precision score : {micro_averaged_precision}")

        # append the result to the end of the log file
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

    else:

        PROBS = np.concatenate(PROBS)
        TARGETS = np.concatenate(TARGETS)

        if args.task_type == 1:
            object_confusion_matrix = torch.zeros(5, 4)
            for i in range(len(PROBS)):
                if PROBS[i] == TARGETS[i, 0]:
                    object_confusion_matrix[TARGETS[i, 0], TARGETS[i, 1]] += 1


            object_confusion_matrix[0:4, :] = object_confusion_matrix[0:4, :] / 12
            object_confusion_matrix[4, :] = object_confusion_matrix[4, :] / 6

            Accuracy = (PROBS == TARGETS[:,0]).mean() * 100.

            content = time.ctime() + ' ' + f'Eval {args.eval}:\nAccuracy : {Accuracy:.5f}\n'

            micro_averaged_precision = metrics.precision_score(TARGETS[:, 0], PROBS, average='micro')

            print(f"\nTask name : {Task_name()}")
            print(f"Micro-Averaged Precision score : {micro_averaged_precision}")

            # append the result to the end of the log file
            print(content)
            with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')
        else:
            Accuracy = (PROBS == np.concatenate(TARGETS)).mean() * 100.

            content = time.ctime() + ' ' + f'Eval {args.eval}:\nAccuracy : {Accuracy:.5f}\n'

            micro_averaged_precision = metrics.precision_score(TARGETS[:, 0], PROBS, average='micro')

            print(f"\nTask name : {Task_name()}")
            print(f"Micro-Averaged Precision score : {micro_averaged_precision}")

            # append the result to the end of the log file
            print(content)
            with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')

    visualization(confusion_matrix)

    if args.task_type == 1:
        visualization(object_confusion_matrix,obj = True)
    #np.save(os.path.join(args.oof_dir, f'{args.kernel_type}_{args.eval}_oof.npy'), dfs['pred'].values)

    ## 결과 csv 저장
    #dfs[['image_name', 'printer', 'target', 'pred']].to_csv(os.path.join(args.oof_dir, f'{args.kernel_type}_{args.eval}_oof.csv'), index=True)

    # 결과 csv 저장
    # dfs[['printer','num','filament','quality','obj_id','model','target', 'pred1','pred2','pred3','pred4','pred5','pred6','pred7','pred8','pred9','pred10','pred11','pred12','pred13','pred14']].to_csv(
    #     os.path.join(args.oof_dir,'fullshot_device_ensemble.csv'), index=True)








if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')
    args = parse_args()
    os.makedirs(args.oof_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    # 네트워크 타입 설정
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

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
