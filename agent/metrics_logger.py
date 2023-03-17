import numpy as np


class MetricsLogger:


    def __init__(self):
        self.eps_metrics = {
                                'loss': [], 'reward': [], 'acc': [], 'f1': [],
                                'val_loss': [], 'val_reward': [], 'val_acc': [], 'val_f1': [],
                            }
        self.init_episode()
    
    def init_episode(self):
        self.cur_ep = {
                        'reward': 0, 'loss': 0, 'loss_length': 0,
                        'val_reward': 0, 'val_loss': 0, 'val_loss_length': 0,
                    }
    
    def log_step(self, reward, loss, prefix=''):
        self.cur_ep[prefix + 'reward'] += reward
        self.cur_ep[prefix + 'loss'] += loss
        if loss:
            self.cur_ep[prefix + 'loss_length'] += 1
        
    def log_episode(self, acc, f1, prefix=''):
        self.eps_metrics[prefix + 'reward'].append(self.cur_ep[prefix + 'reward'])
        self.eps_metrics[prefix + 'acc'].append(acc)
        self.eps_metrics[prefix + 'f1'].append(f1)
        if self.cur_ep[prefix + 'loss_length'] == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.cur_ep[prefix + 'loss'] / self.cur_ep[prefix + 'loss_length'], 5)
        self.eps_metrics[prefix + 'loss'].append(ep_avg_loss)
        self.init_episode()
    
    def get_latest_logs(self):
        latest_logs = {
                        'loss': self.eps_metrics['loss'][-1], 'reward': self.eps_metrics['reward'][-1],\
                        'acc': self.eps_metrics['acc'][-1], 'f1': self.eps_metrics['f1'][-1],
                        'val_loss': self.eps_metrics['val_loss'][-1], 'val_reward': self.eps_metrics['val_reward'][-1],\
                        'val_acc': self.eps_metrics['val_acc'][-1], 'val_f1': self.eps_metrics['val_f1'][-1],      
        }
        return latest_logs

""" 
class MetricsLogger:


    def __init__(self):
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_accs = []
        self.ep_f1s = []

        self.init_episode()
    
    def init_episode(self):
        self.cur_ep_reward = 0
        self.cur_ep_loss = 0
        self.cur_ep_loss_length = 0
    
    def log_step(self, reward, loss, prefix=''):
        self.cur_ep_reward += reward
        self.cur_ep_loss += loss
        if loss:
            self.cur_ep_loss_length += 1
        
    def log_episode(self, acc, f1, prefix=''):
        self.ep_rewards.append(self.cur_ep_reward)
        self.ep_accs.append(acc)
        self.ep_f1s.append(f1)
        if self.cur_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.cur_ep_loss / self.cur_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        logs = {'reward' : self.cur_ep_reward, 'loss' : ep_avg_loss, 'acc' : acc, 'f1' : f1}
        
        
        self.init_episode()
        return logs
    
    def get_latest_logs(self):
        pass
 """