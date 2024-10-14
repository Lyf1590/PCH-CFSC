import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from dset import load_data
import numpy as np
import scipy.io as sio

# 定义全局变量
train_com = None
train_data = None


class Session:
    def __init__(self):
        global train_com, train_data
        self.logger = settings.logger

        if settings.DATASET_NAME == 'flickr':
            # 设置比例0.8 0.5、0.6 0.5 、0.4 0.5、0.2 0.5
            train_com, train_data, _ = load_data(settings.alpha_train, settings.beta_train, settings.alpha_query, settings.beta_query)
            print('settings.DATASET_NAME =', settings.DATASET_NAME)
        elif settings.DATASET_NAME == 'nus-wide':
            train_com, train_data, _ = load_data(settings.alpha_train, settings.beta_train, settings.alpha_query, settings.beta_query)
            print('settings.DATASET_NAME =', settings.DATASET_NAME)
        elif settings.DATASET_NAME == 'coco':
            train_com, train_data, _ = load_data(settings.alpha_train, settings.beta_train, settings.alpha_query, settings.beta_query)
            print('settings.DATASET_NAME =', settings.DATASET_NAME)
        else:
            raise ValueError("Unsupported dataset name")


        self.I_te = train_com['I_te'].numpy().astype(np.float64)
        self.T_te = train_com['T_te'].numpy().astype(np.float64)
        self.L_te = train_com['L_te'].numpy().astype(np.float64)
        self.I_db = train_com['I_db'].numpy().astype(np.float64)
        self.T_db = train_com['T_db'].numpy().astype(np.float64)
        self.L_db = train_com['L_db'].numpy().astype(np.float64)

        self.I_tr = train_com['I_tr']
        self.T_tr = train_com['T_tr']
        self.L_tr = train_com['L_tr'].numpy().astype(np.float64)

        self.L_cat_r = train_com['L_tr']
        print('L_cat_r.shape:', self.L_cat_r.shape)
        self.I_tr = train_data['dual_img']
        print('I_tr.shape:', self.I_tr.shape)
        self.T_tr = train_data['dual_txt']
        self.only_I_tr = train_data['o_img']
        self.only_T_tr = train_data['o_txt']
        print('only_I_tr.shape:', self.only_I_tr.shape)
        miss_txt_zero = np.zeros((self.only_I_tr.shape[0], self.only_T_tr.shape[1]))
        miss_img_zero = np.zeros((self.only_T_tr.shape[0], self.only_I_tr.shape[1]))

        miss_txt_1 = np.full((self.only_I_tr.shape[0], self.only_T_tr.shape[1]), -1)
        miss_img_1 = np.full((self.only_T_tr.shape[0], self.only_I_tr.shape[1]), -1)

        I_cat_r = np.concatenate((self.I_tr, self.only_I_tr, miss_img_zero), axis=0)
        T_cat_r = np.concatenate((self.T_tr, miss_txt_zero, self.only_T_tr), axis=0)

        I_cat_r_1 = np.concatenate((self.I_tr, self.only_I_tr, miss_img_1), axis=0)
        T_cat_r_1 = np.concatenate((self.T_tr, miss_txt_1, self.only_T_tr), axis=0)


        # 保存  801010.mat
        self.logger.info('Dual Complete ratio is %.1f!  Only image ratio is %.2f!  Only text ratio is %.2f!' % (
        settings.alpha_train, (1 - settings.alpha_train) * settings.beta_train, (1 - settings.alpha_train) * settings.beta_train))
        str1 = str(int(settings.alpha_train * 100))
        str2 = str(int((1 - settings.alpha_train) * settings.beta_train * 100))
        str3 = str(int((1 - settings.alpha_train) * settings.beta_train * 100))
        str_name = str1 + str2 + str3

        filename = 'dataset/nus_1/train_%s.mat' % (str_name)
        sio.savemat(filename, {'I_tr': I_cat_r_1, 'T_tr': T_cat_r_1, 'L_tr': self.L_tr,
                                                          'I_te': self.I_te, 'T_te': self.T_te, 'L_te': self.L_te,
                                                          'I_db': self.I_db, 'T_db': self.T_db, 'L_db': self.L_db})


