import torch
import torch.utils.data as data
import settings
import scipy.io as sio
import math
import numpy as np


class MY_DATASET(data.Dataset):
    def __init__(self, img, txt, label):
        self.img = img
        self.txt = txt
        self.label = label

    #  获取数据集的索引
    def __getitem__(self, index):
        img = self.img[index, :]
        txt = self.txt[index, :]
        target = self.label[index, :]
        return img, txt, target, index

    #  获取数据集的大小
    def __len__(self):
        """Return the total number of samples"""
        return len(self.label)


def load_data(alpha_train=1.0, beta_train=0.5, alpha_query=1.0, beta_query=0.5):
    data_set = sio.loadmat(settings.DIR)
    L_tr = torch.Tensor(data_set['L_tr'][:])
    T_tr = torch.Tensor(data_set['T_tr'][:])
    I_tr = torch.Tensor(data_set['I_tr'][:])

    L_db = torch.Tensor(data_set['L_db'][:])
    T_db = torch.Tensor(data_set['T_db'][:])
    I_db = torch.Tensor(data_set['I_db'][:])

    L_te = torch.Tensor(data_set['L_te'][:])
    T_te = torch.Tensor(data_set['T_te'][:])
    I_te = torch.Tensor(data_set['I_te'][:])

    complete_data = {'I_tr': I_tr, 'T_tr': T_tr, 'L_tr': L_tr,
                     'I_db': I_db, 'T_db': T_db, 'L_db': L_db,
                     'I_te': I_te, 'T_te': T_te, 'L_te': L_te}
    print('Train_data：')
    train_missed_data = construct_missed_data(I_tr, T_tr, L_tr, alpha=alpha_train, beta=beta_train)
    print('Query_data：')
    query_missed_data = construct_missed_data(I_te, T_te, L_te, alpha=alpha_query, beta=beta_query)

    return (complete_data, train_missed_data, query_missed_data)


def construct_missed_data(I_tr, T_tr, L_tr, alpha=1.0, beta=0.5):  # 默认是完整的
    # 还是采用
    number = I_tr.shape[0]
    dual_size = math.ceil(number * alpha)
    only_image_size = math.floor((number - dual_size) * beta)
    only_text_size = number - dual_size - only_image_size
    print('Dual size: %d, Only img size: %d, Only txt size: %d' % (dual_size, only_image_size, only_text_size))
    random_idx = np.random.permutation(number)
    dual_index = random_idx[:dual_size]
    only_image_index = random_idx[dual_size:dual_size+only_image_size]
    only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size]
    dual_img = I_tr[dual_index, :]
    dual_txt = T_tr[dual_index, :]
    dual_label = L_tr[dual_index, :]
    only_img = I_tr[only_image_index, :]
    only_img_label = L_tr[only_image_index, :]
    only_txt = T_tr[only_text_index, :]
    only_txt_label = L_tr[only_text_index, :]
    _dict = {'dual_img': dual_img, 'dual_txt': dual_txt, 'dual_label': dual_label, 'o_img': only_img, 'o_txt': only_txt,'o_img_label': only_img_label, 'o_txt_label': only_txt_label}
    return _dict