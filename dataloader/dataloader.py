import torch
import numpy as np
import pandas as pd
import os
from utils.logger import get_logger
LOG = get_logger('data_loader')

class DataLoader:


    def __init__(self, vn30_x_train_np, vn30_y_train_np, vn30_x_test_np, vn30_y_test_np, input_dim):
        self.vn30_x_train_np = vn30_x_train_np
        self.vn30_y_train_np = vn30_y_train_np
        self.vn30_x_test_np = vn30_x_test_np
        self.vn30_y_test_np = vn30_y_test_np
        self.input_dim = input_dim
    
    @classmethod
    def from_json(cls, cfg):
        LOG.info(f'Loading dataset...')
        thesis_dir = cfg.path
        test_day = cfg.test_day
        seq_len = cfg.seq_len

        stock_files = os.listdir(thesis_dir)
        feature_dim = 0
        x_train_dict = {}
        y_train_dict = {}
        x_test_dict = {}
        y_test_dict = {}
        
        for file in stock_files:
            stock = pd.read_csv(thesis_dir + file, index_col='date')            
            y = stock.loc[:, ['price_change', 'label']]
            x = stock.drop(['price_change', 'label'], axis=1)
            if feature_dim == 0:
                feature_dim = x.shape[1]

            mean = x[x.index < test_day].mean(axis=0)
            std = x[x.index < test_day].std(axis=0)
            x = (x - mean) / std

            hist_x = pd.concat([x.shift(i) for i in range((seq_len - 1), -1, -1)], axis=1)
            hist_x.drop(hist_x.head(seq_len - 1).index, inplace=True)
            y.drop(y.head(seq_len - 1).index, inplace=True)

            x_train = hist_x[hist_x.index < test_day]
            x_test = hist_x[hist_x.index >= test_day]

            y_train = y[y.index < test_day]
            y_test = y[y.index >= test_day]

            x_train_dict[file[:-4]] = x_train
            y_train_dict[file[:-4]] = y_train
            x_test_dict[file[:-4]] = x_test
            y_test_dict[file[:-4]] = y_test
            
        vn30_x_train = pd.concat(list(x_train_dict.values()), axis=0).sort_index(axis=0)
        vn30_x_train_np = np.reshape(np.array(vn30_x_train, dtype='float32'), (-1, seq_len, feature_dim))

        vn30_x_test = pd.concat(list(x_test_dict.values()), axis=0).sort_index(axis=0)
        vn30_x_test_np = np.reshape(np.array(vn30_x_test, dtype='float32'), (-1, seq_len, feature_dim))

        vn30_y_train = pd.concat(list(y_train_dict.values()), axis=0).sort_index(axis=0)
        vn30_y_train_np = np.array(vn30_y_train, dtype='float32')
        
        vn30_y_test = pd.concat(list(y_test_dict.values()), axis=0).sort_index(axis=0)
        vn30_y_test_np = np.array(vn30_y_test, dtype='float32')

        input_dim = vn30_x_train_np.shape[1:]
        """ print(vn30_x_train_np.shape)
        print(vn30_x_test_np.shape) """
        return cls(vn30_x_train_np, vn30_y_train_np, vn30_x_test_np, vn30_y_test_np, input_dim)
    
    def get_train(self):
        return self.vn30_x_train_np, self.vn30_y_train_np
    
    def get_test(self):
        return self.vn30_x_test_np, self.vn30_y_test_np
    
    def get_input_dim(self):
        return self.input_dim
    
    def get_3D_train(self):
        "tra du lieu duoi dang channel, height, width"
        return np.expand_dims(self.vn30_x_train_np, 1), self.vn30_y_train_np
    
    def get_3D_test(self):
        "tra du lieu duoi dang channel, height, width"
        return np.expand_dims(self.vn30_x_test_np, 1), self.vn30_y_test_np
    """
    @staticmethod
    def preprocess_data(image, image_size):
        return DataLoader._image_to_tensor(DataLoader._resize_and_bgr2gray(image, image_size), image_size)
    
    @staticmethod
    def _image_to_tensor(image, image_size):
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor.astype(np.float32)
        image_tensor = torch.from_numpy(image_tensor)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            image_tensor = image_tensor.cuda()
        return image_tensor

    @staticmethod
    def _resize_and_bgr2gray(image, image_size):
        image = image[0:288, 0:404]
        image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
        image_data[image_data > 0] = 255
        image_data = np.reshape(image_data, (84, 84, 1))
        return image_data
    
    
    @staticmethod
    def produce_new_state(old_state, new_image, image_size):
        new_image = DataLoader().preprocess_data(new_image, image_size)
        return torch.cat((old_state.squeeze(0)[1:, :, :], new_image)).unsqueeze(0)
    """