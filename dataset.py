import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

'''
Points to note when creating a CSV file for image classification
The following two must be included.

target: class number. ex) {0, 1}
image_name: image file name

'''

def get_df_3dprint(k_fold, data_dir, data_folder, out_dim = 1,out_dim2 = 0, img_type = 'close', task_type = 1,side_task = 0):
    '''

    ##### get DataFrame
    Read the CSV file managed by the database and split it for cross validation

    :param k_fold: k_fold value received from argument
    :param out_dim: Number of network outputs
    :param data_dir: Database folder
    :param data_folder: data folder

    :return:
    :target_idx : positive index number
    '''
    # Read data (pd.read_csv / Save dataFrame)
    if img_type == 'full' or img_type == 'close':
        if task_type == 4: # read csv for device-level identification
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close_device.csv'))
            df_train_full = pd.read_csv(os.path.join(data_dir, data_folder, 'train_full_device.csv'))
        elif task_type == 5:
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close_reprint.csv'))
            df_train_full = pd.read_csv(os.path.join(data_dir, data_folder, 'train_full_reprint.csv'))
        else: # read csv with full list of image
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close.csv'))
            df_train_full = pd.read_csv(os.path.join(data_dir, data_folder, 'train_full.csv'))

        df_train_close['filepath'] = df_train_close['image_name'].apply(
            lambda x: os.path.join(data_dir, f'{data_folder}train_close', x))  # f'{x}.jpg'

        df_train_full['filepath'] = df_train_full['image_name'].apply(
            lambda x: os.path.join(data_dir, f'{data_folder}train_full', x))  # f'{x}.jpg'

        # Original data=0, Meta data=1
        df_train_close['is_ext'] = 0
        # Original data=0, Meta data=1
        df_train_full['is_ext'] = 0

    elif img_type == 'both':
        if task_type == 4:
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close_device.csv'))
        elif task_type == 5:
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close_reprint.csv'))
        else:
            df_train_close = pd.read_csv(os.path.join(data_dir, data_folder, 'train_close.csv'))

        df_train_full = pd.read_csv(os.path.join(data_dir, data_folder, 'train_full.csv'))

        df_train_close['filepath'] = df_train_close['image_name'].apply(
            lambda x: os.path.join(data_dir, f'{data_folder}train_close', x))  # f'{x}.jpg'
        df_train_full['filepath'] = df_train_full['image_name'].apply(
            lambda x: os.path.join(data_dir, f'{data_folder}train_full', x))  # f'{x}.jpg'


    # test data
    df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'


    if task_type == 1: # printer task
        # CSV standard [210F : 0, 320C : 1, Finder : 2, Method_X : 3, Replicator : 4]
        target2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.printer.unique()))}
        df_train_close['target'] = df_train_close['printer'].map(target2idx_close)
        target2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.printer.unique()))}
        df_train_full['target'] = df_train_full['printer'].map(target2idx_full)
    elif task_type == 2: # filament task
        # CSV standard  [ABS  : 0, ASA  : 1, NYLON  : 2, PETG : 3, PLA : 4, PLA+  : 5, ToughPLA  :6]
        target2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.filament.unique()))}
        df_train_close['target'] = df_train_close['filament'].map(target2idx_close)
        target2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.filament.unique()))}
        df_train_full['target'] = df_train_full['filament'].map(target2idx_full)
    elif task_type == 3: # quality task
        # CSV standard  [HQ  : 0, MQ  : 1, LQ  : 2]
        target2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.quality.unique()))}
        df_train_close['target'] = df_train_close['quality'].map(target2idx_close)
        target2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.quality.unique()))}
        df_train_full['target'] = df_train_full['quality'].map(target2idx_full)
    elif task_type == 4: # device task
        # CSV standard  [210F_1 ~ 4 : 0 ~ 3 , 320C_1 ~ 4 : 4 ~ 7, Finder_1 ~ 4 : 8 ~ 11, Method_X : 12, Replicator : 13]
        target2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.num.unique()))}
        df_train_close['target'] = df_train_close['num'].map(target2idx_close)
        target2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.num.unique()))}
        df_train_full['target'] = df_train_full['num'].map(target2idx_full)
    else: # reprint task
        # CSV standard  [reprint : 0, not reprint  : 1]
        target2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.reprint.unique()))}
        df_train_close['target'] = df_train_close['reprint'].map(target2idx_close)
        target2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.reprint.unique()))}
        df_train_full['target'] = df_train_full['reprint'].map(target2idx_full)


    if side_task == 1: # side-task : printer task
        target2idx2_close = {d: idx for idx, d in enumerate(sorted(df_train_close.printer.unique()))}
        df_train_close['target2'] = df_train_close['printer'].map(target2idx2_close)
        target2idx2_full = {d: idx for idx, d in enumerate(sorted(df_train_full.printer.unique()))}
        df_train_full['target2'] = df_train_full['printer'].map(target2idx2_full)
    elif side_task == 2: # side-task : filament task
        target2idx2_close = {d: idx for idx, d in enumerate(sorted(df_train_close.filament.unique()))}
        df_train_close['target2'] = df_train_close['filament'].map(target2idx2_close)
        target2idx2_full = {d: idx for idx, d in enumerate(sorted(df_train_full.filament.unique()))}
        df_train_full['target2'] = df_train_full['filament'].map(target2idx2_full)
    elif side_task == 3: # side-task : quality task
        target2idx2_close = {d: idx for idx, d in enumerate(sorted(df_train_close.quality.unique()))}
        df_train_close['target2'] = df_train_close['quality'].map(target2idx2_close)
        target2idx2_full = {d: idx for idx, d in enumerate(sorted(df_train_full.quality.unique()))}
        df_train_full['target2'] = df_train_full['quality'].map(target2idx2_full)
    elif side_task == 4: # side-task : device task
        target2idx2_close = {d: idx for idx, d in enumerate(sorted(df_train_close.num.unique()))}
        df_train_close['target2'] = df_train_close['num'].map(target2idx2_close)
        target2idx2_full = {d: idx for idx, d in enumerate(sorted(df_train_full.num.unique()))}
        df_train_full['target2'] = df_train_full['num'].map(target2idx2_full)
    elif side_task == 5: # side-task : reprint task
        target2idx2_close = {d: idx for idx, d in enumerate(sorted(df_train_close.reprint.unique()))}
        df_train_close['target2'] = df_train_close['reprint'].map(target2idx2_close)
        target2idx2_full = {d: idx for idx, d in enumerate(sorted(df_train_full.reprint.unique()))}
        df_train_full['target2'] = df_train_full['reprint'].map(target2idx2_full)


    printer2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.printer.unique()))}
    printer2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.printer.unique()))}
    model2idx_close = {d: idx for idx, d in enumerate(sorted(df_train_close.model.unique()))}
    model2idx_full = {d: idx for idx, d in enumerate(sorted(df_train_full.model.unique()))}


    df_train_close['printer_id'] = df_train_close['printer'].map(printer2idx_close)
    df_train_full['printer_id'] = df_train_full['printer'].map(printer2idx_full)
    df_train_close['model_id'] = df_train_close['model'].map(model2idx_close)
    df_train_full['model_id'] = df_train_full['model'].map(model2idx_full)

    if img_type == 'both':
        return df_train_close, df_train_full, df_test
    elif img_type == 'full':
        return df_train_full, df_test
    else:
        return df_train_close, df_test

class MMC_ClassificationDataset(Dataset):
    '''
    MMC_ClassificationDataset 클래스
    Dataset class for image classification
        class Dataset_def_name(Dataset):
            def __init__(self, csv, mode, meta_features, transform=None):
                # Dataset initialization

            def __len__(self):
                # return dataset length
                return self.csv.shape[0]

            def __getitem__(self, index):
                # Returns the image corresponding to the index
    '''

    def __init__(self, csv, mode, csv2 = None, transform=None, task_type = 1, side_task = 0,img_type = None):
        self.csv = csv.reset_index(drop=True)
        if csv2 is not None:
            self.csv2 = csv2.reset_index(drop=True)
        else:
            self.csv2 = None
        self.mode = mode # train / valid
        self.transform = transform
        self.task_type = task_type
        self.side_task = side_task
        self.img_type = img_type

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        # Extraction target and object id
        if self.csv2 is not None:
            if self.side_task:
                positive_rand_filepath = random.choice([id for id in self.csv2.loc[self.csv2['target2'] == row.target2].filepath])
            else:
                if self.mode == 'valid':
                    positive_rand_filepath = random.choice([id for id in self.csv2.loc[self.csv2['obj_id'] == row.obj_id].filepath])
                else:
                    positive_rand_filepath = random.choice(
                        [id for id in self.csv2.loc[self.csv2['target'] == row.target].filepath])
            close_image = self._read_image_row(row.filepath)
            full_image = self._read_image_row(positive_rand_filepath) # bring full image random path

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.csv2 is None:
            if self.transform is not None:
                res = self.transform(image=image)
                image = res['image'].astype(np.float32)
            else:
                image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.img_type == 'both':
            data = (close_image, full_image)
            obj_id = torch.tensor(row['obj_id'])
        else:
            data = torch.tensor(image).float()
            obj_id = torch.tensor(row['obj_id'])

        if self.mode == 'valid':
            # validation multi-task multi-modal
            if self.img_type == 'both' and self.side_task:
                if self.task_type == 1:
                    return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                        self.csv.iloc[index].target2).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].model_id).long()
                else:
                    return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                        self.csv.iloc[index].target2).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].printer_id).long()
            # validation single-task multi-modal
            elif self.img_type == 'both' and self.side_task == False:
                if self.task_type == 1:
                    return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].model_id).long()
                else:
                    return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].printer_id).long()
            # validation single-task multi-modal
            elif self.side_task:
                if self.task_type == 1:
                    return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                        self.csv.iloc[index].target2).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].model_id).long()
                else:
                    return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                        self.csv.iloc[index].target2).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].printer_id).long()
            # validation single-task single-modal
            else:
                if self.task_type == 1:
                    return data, torch.tensor(self.csv.iloc[index].target).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].model_id).long()
                else:
                    return data, torch.tensor(self.csv.iloc[index].target).long(), obj_id, torch.tensor(
                        self.csv.iloc[index].printer_id).long()


        else:
            # train multi-task multi-modal
            if self.img_type == 'both' and self.side_task:
                return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(
                    self.csv.iloc[index].target2).long()
            # train single-task multi-modal
            elif self.img_type == 'both' and self.side_task == False:
                return close_image, full_image, torch.tensor(self.csv.iloc[index].target).long()
            # train multi-task single-modal
            elif self.side_task:
                return data, torch.tensor(self.csv.iloc[index].target).long(),torch.tensor(self.csv.iloc[index].target2).long()
            # train single-task single-modal
            else:
                return data, torch.tensor(self.csv.iloc[index].target).long()

    def _read_image_row(self, filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Apply image tranform
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).float()

        return image

def get_transforms(image_size):
    '''
    Use albumentations library
    https://github.com/albumentations-team/albumentations

    TODO: FAST AUTO AUGMENT
    https://github.com/kakaobrain/fast-autoaugment
    DATASET의 AUGMENT POLICY를 탐색해주는 알고리즘

    TODO: Unsupervised Data Augmentation for Consistency Training
    https://github.com/google-research/uda

    TODO: Cutmix vs Mixup vs Gridmask vs Cutout
    https://www.kaggle.com/saife245/cutmix-vs-mixup-vs-gridmask-vs-cutout

    '''

    transforms_train = albumentations.Compose([
        #albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.MedianBlur(blur_limit=3, p=0.75),
        albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

        albumentations.CLAHE(clip_limit=4.0, p=0.4), # 1차 p = 0.7
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val
