import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix


class TrendEnv(gym.Env):

    def __init__(self, x_np, y_np, mode): 
        self.Env_data = x_np
        self.Answer = y_np
        self.game_len = self.Env_data.shape[0]

        self.action_space = spaces.Discrete(2)
        self.step_ind = 0
        self.y_pred = []
        self.mode = mode

    def seed(self, seed=5):
        #self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.y_pred.append(action)
        price_change_idx, label_idx = 0, 1
        
        if self.mode == 'dynamic':
            reward = np.abs(self.Answer[self.step_ind, price_change_idx]) if action == int(self.Answer[self.step_ind, label_idx])\
                     else -np.abs(self.Answer[self.step_ind, price_change_idx]) 
        else: 
            reward = 1 if action == int(self.Answer[self.step_ind, label_idx]) else -1

        done = 0
        info = {}
        state = None
        if self.step_ind == self.game_len - 1:
            info['acc'], info['f1'] = self.return_metrics(np.array(self.y_pred), self.Answer[:, label_idx])
            done = 1
            state = self.reset()
        else:
            self.step_ind += 1
            state = self.Env_data[self.step_ind]
        
        return state, reward, done, info

    def return_metrics(self, y_pre, y_true):
        cf_matrix = confusion_matrix(y_true, y_pre)
        cf_matrix_np = np.array(cf_matrix, dtype='float')
        TP = cf_matrix_np[1][1]
        TN = cf_matrix_np[0][0]
        FN = cf_matrix_np[1][0]
        FP = cf_matrix_np[0][1]
        TPrate = TP / (TP + FN) 
        TNrate = TN / (TN + FP)  
        FPrate = FP / (TN + FP)  
        FNrate = FN / (TP + FN)  
        PPvalue = TP / (TP + FP)  
        NPvalue = TN / (TN + FN) 

        Accuracy = (TP + TN)/ (TP + TN + FN + FP)
        Recall = TPrate
        Precision = PPvalue 
        F1 = 2 * Recall * Precision / (Recall + Precision)
        return Accuracy, F1

    def reset(self):
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.step_ind]