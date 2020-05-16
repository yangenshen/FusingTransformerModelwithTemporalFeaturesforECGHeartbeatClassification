# -*- coding: utf-8 -*-
import time
import sklearn
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import numpy as np
import pandas as pd
from model import Transformer
from optim import ScheduledOptim
from dataset import SignalDataset
from config import *
from FocalLoss import FocalLoss
from entropy import *
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from roc import plot_roc
from imblearn.over_sampling import SMOTE
import time
import os

FL = FocalLoss(class_num=4, gamma=1.5, average=False)


def cal_loss(pred, label, device):
    cnt_per_class = np.zeros(4)
    loss = FL(pred, label, device)

    loss = F.cross_entropy(pred, label, reduction='sum')
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    cnt_per_class = [cnt_per_class[j] + pred.eq(j).sum().item() for j in range(class_num)]
    return loss, n_correct, cnt_per_class


def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)
    # total_true = np.array([17703,   491,  1357,   159])
    # special acc, abnormal inlcuded only
    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1:class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, list(pre_i), list(rec_i), list(F1_i)


def train_epoch(train_loader, device, model, optimizer, total_num):
    all_labels = []
    all_res = []
    model.train()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    for batch in tqdm(train_loader, mininterval=0.5, desc='- (Training)  ', leave=False):
        sig, fea_plus, label, = map(lambda x: x.to(device), batch)
        # forward
        optimizer.zero_grad()
        pred = model(sig, fea_plus)
        all_labels.extend(label.cpu().numpy())
        all_res.extend(pred.max(1)[1].cpu().numpy())
        # backward
        loss, n_correct, cnt = cal_loss(pred, label, device)
        loss.backward()
        # update
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_correct += n_correct
        cnt_per_class += cnt
    # cm = confusion_matrix(all_labels, all_res)
    # print(cm)
    # acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    # print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    # print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    # print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    # print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cnt_per_class


def eval_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
            sig, fea_plus, label, = map(lambda x: x.to(device), batch)
            # forward
            pred = model(sig, fea_plus)  # emd.contiguous().view(len(label), fea_num, -1)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt
    cm = confusion_matrix(all_labels, all_res)
    # print(cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cnt_per_class, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4


def test_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pres = []
    all_recs = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
            sig, fea_plus, label, = map(lambda x: x.to(device), batch)
            # forward
            pred = model(sig, fea_plus)  # emd.contiguous().view(len(label), fea_num, -1)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt

    # np.savetxt('all_pres.txt',all_pres)
    # np.savetxt('all_recs.txt', all_recs)
    np.savetxt('all_pred.txt',all_pred)
    np.savetxt('all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    plot_roc(all_labels,all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print(cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))


if __name__ == '__main__':

    model_name = 'transform.chkpt'
    train_file = 'data/train_features.txt'
    valid_file = 'data/valid_features.txt'
    raw_train = pd.read_csv(train_file, header=None).values
    raw_valid = pd.read_csv(valid_file, header=None).values
    whole_data = np.vstack((raw_train, raw_valid))

    # a, b = SMOTE().fit_sample(whole_data[:, 1:], whole_data[:, 0])
    # whole_data = np.c_[b, a]

    for r in range(10):
        time_start_i = time.time()
        raw_train, raw_valid, _, _ = train_test_split(whole_data, list(whole_data[:, 0]), test_size=0.3,
                                                          random_state=r,stratify=list(whole_data[:, 0]))
        raw_valid, raw_test, _, _ = train_test_split(raw_valid, list(raw_valid[:, 0]), test_size=0.6666,
                                                     random_state=r,stratify=list(raw_valid[:, 0]))

        if torch.cuda.is_available():
            device = torch.device('cuda:3')
        else:
            device = torch.device('cpu')

        train_data = SignalDataset(raw_train)
        valid_data = SignalDataset(raw_valid)
        test_data = SignalDataset(raw_test)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=2,
                                 shuffle=True)

        model = Transformer(device=device, d_feature=train_data.sig_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num)

        model = model.to(device)

        optimizer = ScheduledOptim(
            Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 betas=(0.9, 0.98), eps=1e-09), d_model, warm_steps)
        train_accs = []
        valid_accs = []
        eva_indis = []
        train_losses = []
        valid_losses = []
        for epoch_i in range(epoch):
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_acc, cnt = train_epoch(train_loader, device, model, optimizer, train_data.__len__())
            print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                     elapse=(time.time() - start) / 60))
            print(cnt)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            start = time.time()
            valid_loss, valid_acc, cnt, eva_indi = eval_epoch(valid_loader, device, model, valid_data.__len__())
            print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                     elapse=(time.time() - start) / 60))
            print(cnt)
            valid_accs.append(valid_acc)
            eva_indis.append(eva_indi)
            valid_losses.append(valid_loss)
            model_state_dict = model.state_dict()

            checkpoint = {
                'model': model_state_dict,
                'config_file': 'config',
                'epoch': epoch_i}

            if eva_indi >= max(eva_indis):
                torch.save(checkpoint, str(r)+model_name)
                print('    - [Info] The checkpoint file has been updated.')

        print('——ALL DONE!——')
        time_consume = (time.time() - time_start_i)
        print('total ' + str(time_consume) + 'seconds')
        plt.plot(valid_losses)
        plt.xlabel('epoch')
        plt.ylim([0.0, 0.5])
        plt.ylabel('valid loss')
        plt.title('loss change curve')
        plt.show()
        # pre = time.strft  ime("%Y-%m-%d_%H:%M:%S", time.localtime())
        # config = [batch_size, d_model, num_layers, num_heads, class_num, d_inner, dropout]
        # np.save(pre + '_config.npy', config)
        # np.save(pre + '_train_accs.npy', train_accs)
        # np.save(pre + '_valid_accs.npy', valid_accs)
        # np.save(pre + '_train_losses.npy', train_losses)
        # np.save(pre + '_valid_losses.npy', valid_losses)
        test_model_name = str(r) + model_name
        model = Transformer(device=device, d_feature=test_data.sig_len, d_model=d_model, d_inner=d_inner,
                            n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
                            class_num=class_num)
        chkpoint = torch.load(test_model_name, map_location='cuda:3')
        model.load_state_dict(chkpoint['model'])
        model = model.to(device)
        test_epoch(test_loader, device, model, test_data.__len__())

    # models = []
    # for r in range(10):
    #     test_model_name = str(r) + model_name
    #     model = Transformer(device=device, d_feature=test_data.sig_len, d_model=d_model, d_inner=d_inner,
    #                         n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
    #                         class_num=class_num)
    #     chkpoint = torch.load(test_model_name)
    #     model.load_state_dict(chkpoint['model'])
    #     model = model.to(device)
    #     models.append(model)
    # voting_epoch(test_loader, device, models, test_data.__len__())
