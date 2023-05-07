from configs.config import CFG
from utils.config import Config
from dataloader.dataloader import DataLoader
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import pandas as pd

def plot_hist_train(vn30_dict, vn30_tickets, trading_days, start_day_ind, end_day_ind, figs_path):
    for ticket in vn30_tickets:
        close_change = vn30_dict[ticket].loc[trading_days[start_day_ind: end_day_ind], 'close'] - 1
        fig, axes = plt.subplots(layout='constrained')

        axes.hist(close_change, bins='auto', histtype='bar', density=True)
        axes.set_title(ticket)

        fig.savefig(figs_path + f'{ticket}.png')

def plot_hist_test(vn30_dict, vn30_tickets, trading_days, start_day_ind, end_day_ind, figs_path):
    vn30_close_change = np.array([vn30_dict[ticket].loc[trading_days[start_day_ind: end_day_ind], 'close'] - 1 for ticket in vn30_tickets]).reshape(-1)
    fig, axes = plt.subplots(layout='constrained')

    axes.hist(vn30_close_change, bins='auto', histtype='bar', density=True)
    axes.set_title('VN30')

    fig.savefig(figs_path + f'VN30.png')
    plt.close()

def save_df_dist(vn30_dict, vn30_tickets, trading_days, train_start_day_ind, train_end_day_ind, test_start_day_ind, test_end_day_ind,path, label_thres):
    
    labels_dist_df = pd.DataFrame()
    for ticket in vn30_tickets:
        close_changes = (vn30_dict[ticket].loc[trading_days[train_start_day_ind: train_end_day_ind], 'close'] - 1).to_numpy()
        labels = np.where(close_changes > label_thres, 2,\
                        np.where(close_changes < -label_thres, 0, 1)) 
        labels_dist = pd.Series(labels).value_counts(normalize=True)
        labels_dist_df = labels_dist_df.append(labels_dist, ignore_index=True)
        
    
    vn30_close_changes = pd.concat([vn30_dict[ticket].loc[trading_days[test_start_day_ind: test_end_day_ind], 'close'] - 1 for ticket in vn30_tickets], axis=0).to_numpy()
    vn30_labels = np.where(vn30_close_changes > label_thres, 2,\
                        np.where(vn30_close_changes < -label_thres, 0, 1)) 
    vn30_labels_dist = pd.Series(vn30_labels).value_counts(normalize=True)
    labels_dist_df = labels_dist_df.append(vn30_labels_dist, ignore_index=True)
    vn30_tickets.append('vn30_test')
    labels_dist_df.index = vn30_tickets
    labels_dist_df.to_csv(path + f'{label_thres}.csv')
        
        

if __name__ == "__main__":
    config = Config.from_json(CFG)
    data_loader = DataLoader.from_json(config.data)
    label_thres = 0.005
    figs_path = './' + 'figs/'
    dist_path = f'./debug/dist/'
    os.makedirs(dist_path, exist_ok=True)
    """ 
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cf_mt = confusion_matrix(y_true, y_pred, normalize='all')
    true_ind_len, prediction_ind_len = cf_mt.shape
    cf_mt_dict = {f'true-{true_ind}_prediction-{prediction_ind}' : cf_mt[true_ind][prediction_ind] for true_ind in range(true_ind_len) for prediction_ind in range(prediction_ind_len)}
    print(cf_mt)
    print(cf_mt.shape)
    print(cf_mt_dict)
    exit() """
    """ json_object = json.dumps(CFG)
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
    exit() """

    vn30_dict = data_loader.get_vn30_df_dict()
    vn30_tickets = data_loader.get_tickets()
    trading_days = data_loader.get_trading_days()
    train_start_day_ind, train_end_day_ind, test_start_day_ind, test_end_day_ind = data_loader.get_days_ind_train_test_split(data_loader.get_test_day_ind())

    #plot_hist_train(vn30_dict, vn30_tickets, trading_days, train_start_day_ind, train_end_day_ind, figs_path)
    #plot_hist_test(vn30_dict, vn30_tickets, trading_days, test_start_day_ind, test_end_day_ind, figs_path)
    save_df_dist(vn30_dict, vn30_tickets, trading_days, train_start_day_ind, train_end_day_ind, test_start_day_ind, test_end_day_ind, dist_path, label_thres)