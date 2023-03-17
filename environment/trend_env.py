import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import confusion_matrix


def return_metrics(y_pre, y_true):
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

class TrendEnv(gym.Env):

    def __init__(self, x_np, y_np_dict, mode): 
        self.env_data = x_np
        self.label = y_np_dict['label']
        self.price_change = y_np_dict['price_change']
        self.game_len = self.env_data.shape[0]
        
        self.step_ind = 0
        self.prediction = []
        self.mode = mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.NINF, high=np.PINF, shape=self.env_data.shape[1:], dtype='float')
        

    """ def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] """

    def step(self, action):
        self.prediction.append(action)
        if self.mode == 'dynamic':
            reward = np.abs(self.price_change[self.step_ind]) if action == int(self.label[self.step_ind])\
                     else -np.abs(self.price_change[self.step_ind]) 
        else: 
            reward = 1 if action == int(self.label[self.step_ind]) else -1

        done = 0
        info = {}
        state = None
        if self.step_ind == self.game_len - 1:
            info['acc'], info['f1'] = return_metrics(np.array(self.prediction), self.label)
            done = 1
            state = self.reset()
        else:
            self.step_ind += 1
            state = self.env_data[self.step_ind]
        
        return state, reward, done, info

    def reset(self):
        self.step_ind = 0
        self.prediction = []
        return self.env_data[self.step_ind]