import torch
import numpy as np
import pandas as pd
import os
import talib
from talib.abstract import *
from utils.logger import get_logger
LOG = get_logger('data_loader')


def add_features(stock_df):
    #add indicators
    """ inputs = {
        'open': stock_df['open'].to_numpy(dtype='float'), 
        'high': stock_df['high'].to_numpy(dtype='float'), 
        'low': stock_df['low'].to_numpy(dtype='float'), 
        'close': stock_df['close'].to_numpy(dtype='float'), 
        'volume': stock_df['volume'].to_numpy(dtype='float'), 
    }
    stock_df['macd'], _, _ = MACD(inputs)
    stock_df['boll_ub'], _, stock_df['boll_lb'] = BBANDS(inputs)
    stock_df['rsi_30'] = RSI(inputs, timeperiod=30)
    stock_df['cci_30'] = CCI(inputs, timeperiod=30)
    stock_df['dx_30'] = DX(inputs, timeperiod=30)
    stock_df['close_sma_30'] = SMA(inputs, timeperiod=30)
    stock_df['close_sma_60'] = SMA(inputs, timeperiod=60) """

    #add output
    stock_df['price_change'] = stock_df['close'].shift(-1) / stock_df['close'] - 1
    stock_df['label'] = np.where(np.array(stock_df['price_change']) > 0, 1, 0)

    #remove na rows
    stock_df.dropna(axis=0, how='any', inplace=True)
    return stock_df

class DataLoader:


    def __init__(self, vn30_x_train_np, vn30_y_train_np_dict, vn30_x_val_np, vn30_y_val_np_dict, vn30_x_test_np, vn30_y_test_np_dict, input_dim):
        self.vn30_x_train_np = vn30_x_train_np
        self.vn30_y_train_np_dict = vn30_y_train_np_dict
        self.vn30_x_val_np = vn30_x_val_np
        self.vn30_y_val_np_dict = vn30_y_val_np_dict
        self.vn30_x_test_np = vn30_x_test_np
        self.vn30_y_test_np_dict = vn30_y_test_np_dict
        self.input_dim = input_dim
    
    @classmethod
    def from_json(cls, cfg):
        LOG.info(f'Loading dataset...')
        thesis_dir = cfg.path
        test_day = cfg.test_day
        val_day = cfg.val_day
        seq_len = cfg.seq_len

        stock_files = os.listdir(thesis_dir)
        x_train_dict = {}
        y_train_dict = {}
        x_val_dict = {}
        y_val_dict = {}
        x_test_dict = {}
        y_test_dict = {}
        
        for file in stock_files:
            stock_df = pd.read_csv(thesis_dir + file, index_col='date')
            stock_df = add_features(stock_df)

            y = stock_df.loc[:, ['price_change', 'label']]
            x = stock_df.drop(['price_change', 'label'], axis=1)

            mean = x[x.index < val_day].mean(axis=0)
            std = x[x.index < val_day].std(axis=0)
            x = (x - mean) / std

            hist_x = pd.concat([x.shift(i) for i in range((seq_len - 1), -1, -1)], axis=1)
            hist_x.drop(hist_x.head(seq_len - 1).index, inplace=True)
            y.drop(y.head(seq_len - 1).index, inplace=True) 

            x_train = hist_x[hist_x.index < val_day]
            x_val = hist_x[(hist_x.index >= val_day) & (hist_x.index < test_day)]
            x_test = hist_x[hist_x.index >= test_day]

            y_train = y[y.index < val_day]
            y_val = y[(y.index >= val_day) & (y.index < test_day)]
            y_test = y[y.index >= test_day]

            x_train_dict[file[:-4]] = x_train
            y_train_dict[file[:-4]] = y_train
            x_val_dict[file[:-4]] = x_val
            y_val_dict[file[:-4]] = y_val
            x_test_dict[file[:-4]] = x_test
            y_test_dict[file[:-4]] = y_test 
            
        vn30_x_train = pd.concat(list(x_train_dict.values()), axis=0).sort_index(axis=0)
        vn30_x_train_np = np.reshape(np.array(vn30_x_train, dtype='float32'), (vn30_x_train.shape[0], seq_len, -1))

        vn30_x_val = pd.concat(list(x_val_dict.values()), axis=0).sort_index(axis=0)
        vn30_x_val_np = np.reshape(np.array(vn30_x_val, dtype='float32'), (vn30_x_val.shape[0], seq_len, -1))

        vn30_x_test = pd.concat(list(x_test_dict.values()), axis=0).sort_index(axis=0)
        vn30_x_test_np = np.reshape(np.array(vn30_x_test, dtype='float32'), (vn30_x_test.shape[0], seq_len, -1))

        vn30_y_train = pd.concat(list(y_train_dict.values()), axis=0).sort_index(axis=0)
        vn30_y_train_np_dict = {'price_change' : np.array(vn30_y_train['price_change'], dtype='float32'),\
                                 'label' : np.array(vn30_y_train['label'], dtype='int32')}
        
        vn30_y_val = pd.concat(list(y_val_dict.values()), axis=0).sort_index(axis=0)
        vn30_y_val_np_dict = {'price_change' : np.array(vn30_y_val['price_change'], dtype='float32'),\
                                 'label' : np.array(vn30_y_val['label'], dtype='int32')}
        
        vn30_y_test = pd.concat(list(y_test_dict.values()), axis=0).sort_index(axis=0)
        vn30_y_test_np_dict = {'price_change' : np.array(vn30_y_test['price_change'], dtype='float32'),\
                                 'label' : np.array(vn30_y_test['label'], dtype='int32')}

        input_dim = vn30_x_train_np.shape[1:]
        """ 
        print(vn30_x_val_np.shape)
        print(vn30_y_val_np_dict)
         """
        LOG.info(f'Loading finished')
        
        return cls(vn30_x_train_np, vn30_y_train_np_dict, vn30_x_val_np, vn30_y_val_np_dict, vn30_x_test_np, vn30_y_test_np_dict, input_dim)
    
    def get_train(self):
        return self.vn30_x_train_np, self.vn30_y_train_np_dict
    
    def get_val(self):
        return self.vn30_x_val_np, self.vn30_y_val_np_dict
    
    def get_test(self):
        return self.vn30_x_test_np, self.vn30_y_test_np_dict
    
    def get_3D_train(self):
        "tra du lieu duoi dang channel, height, width"
        return np.expand_dims(self.vn30_x_train_np, axis=1), self.vn30_y_train_np_dict
    
    def get_3D_val(self):
        return np.expand_dims(self.vn30_x_val_np, axis=1), self.vn30_y_val_np_dict
    
    def get_3D_test(self):
        "tra du lieu duoi dang channel, height, width"
        return np.expand_dims(self.vn30_x_test_np, axis=1), self.vn30_y_test_np_dict

    def get_input_dim(self):
        return self.input_dim