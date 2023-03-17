from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

class CallBack():
    def __init__(self): 
        pass

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs=None): 
        pass

    def on_train_end(self, logs=None): 
        pass

    def on_validation_begin(self, logs=None):
        pass

    def on_validation_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass
        
    def on_epoch_begin(self, epoch, logs=None): 
        pass

    def on_epoch_end(self, epoch, logs=None): 
        pass

    

class AgentChechPoint(CallBack):
    def __init__(self, filepath, monitor): 
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor

    def on_train_begin(self, logs=None): 
        self.best_policy_weight = None
        self.best_target_weight = None
        self.exploration_rate = None
        self.best_monitor_var = np.Inf

    def on_train_end(self, logs=None): 
        torch.save(
            dict(policy=self.best_policy_weight, target=self.best_target_weight, exploration_rate=self.exploration_rate),
            self.filepath + 'best.chkpt',
        )
        
    def on_epoch_end(self, epoch, logs=None): 
        #print(epoch)
        cur_monitor_var = logs.get(self.monitor)
        if cur_monitor_var < self.best_monitor_var:
            self.best_policy_weight = self.trainer.agent.policy_net.state_dict()
            self.best_target_weight = self.trainer.agent.target_net.state_dict()
            self.exploration_rate = self.trainer.agent.exploration_rate
            self.best_monitor_var = cur_monitor_var
        torch.save(
            dict(policy=self.best_policy_weight, target=self.best_target_weight, exploration_rate=self.exploration_rate),
            self.filepath + f'episode-{epoch}.chkpt',
        )

class TensorBoard(CallBack):
    def __init__(self, log_dir): 
        super().__init__()
        self.log_dir = log_dir

    def on_train_begin(self, logs=None): 
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None): 
        metrics = self.trainer.metrics
        for metric in metrics:
            self.writer.add_scalars(metric, {'Train': logs[metric], 'Validation': logs['val_' + metric]}, epoch)


class EarlyStopping(CallBack):
    def __init__(self, monitor, min_delta=0, patience=0): 
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_monitor_var = np.Inf

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Episode %05d: early stopping" % (self.stopped_epoch))
    
    def on_epoch_end(self, epoch, logs=None): 
        print('------------------')
        print(f"Episode {epoch}:")
        print(f"Train: Loss {logs['loss']}, Reward {logs['reward']}, Accuracy {logs['acc']}, F1-score {logs['f1']}")
        print(f"Validation: Loss {logs['val_loss']}, Reward {logs['val_reward']}, Accuracy {logs['val_acc']}, F1-score {logs['val_f1']}")
        cur_monitor_var = logs.get(self.monitor)
        if (cur_monitor_var - self.best_monitor_var) < self.min_delta:
            self.best_monitor_var = cur_monitor_var
            self.wait = 0
        else: 
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.should_stop = True


class CallBackList(CallBack):
    def __init__(self, callbacks): 
        super().__init__()
        self.callbacks = callbacks
    
    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs=None): 
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None): 
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_validation_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_end(logs)

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_epoch_begin(self, epoch, logs=None): 
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None): 
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)