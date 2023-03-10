import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.logger import get_logger
LOG = get_logger('metrics_logger')


class MetricsLogger:


    def __init__(self, logs_dir):
        self.writer = SummaryWriter(logs_dir)

        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_accs = []
        self.ep_f1s = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_accs = []
        self.moving_avg_ep_f1s = []

        self.init_episode()
    
    def init_episode(self):
        self.cur_ep_reward = 0
        self.cur_ep_loss = 0
        self.cur_ep_loss_length = 0
    
    def log_step(self, reward, loss):
        self.cur_ep_reward += reward
        self.cur_ep_loss += loss
        if loss:
            self.cur_ep_loss_length += 1
        
    def log_episode(self, acc, f1):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_accs.append(acc)
        self.ep_f1s.append(f1)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss / self.cur_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        print('---------------------')
        print(f'Reward: {self.cur_ep_reward}, Loss: {ep_avg_loss}, Accuracy: {acc}, F1-score: {f1}')
        
        self.init_episode()

    def record(self, episode):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_acc = np.round(np.mean(self.ep_accs[-100:]), 3)
        mean_ep_f1 = np.round(np.mean(self.ep_f1s[-100:]), 3)

        self.writer.add_scalar('Reward', mean_ep_reward, episode)
        self.writer.add_scalar('Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Accuracy', mean_ep_acc, episode)
        self.writer.add_scalar('F1-score', mean_ep_f1, episode)
