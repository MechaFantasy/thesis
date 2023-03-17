# internal
from utils.logger import get_logger

# external
import torch
LOG = get_logger('trainer')


class TrendAgentTrainer:
    
    def __init__(self, agent, env, val_env, metrics, logger, callbacks):
        self.agent = agent
        self.env = env
        self.val_env = val_env
        self.metrics = metrics
        self.logger = logger 
        self.callbacks = callbacks
    
        self.should_stop = False

        self.callbacks.set_trainer(self)
    
    def train(self):
        LOG.info('Training started')
        self.callbacks.on_train_begin()
        self.callbacks.on_validation_begin()

        for episode in range(self.agent.start_from_episode, self.agent.episodes + 1):
            self.callbacks.on_epoch_begin(episode)

            state = self.env.reset()
            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.cache(state, next_state, action, reward, done)
                loss = self.agent.learn()
                state = next_state

                self.logger.log_step(reward, loss)
                if done == 1:
                    self.logger.log_episode(info['acc'], info['f1'])
                    break

            val_state = self.val_env.reset()
            with torch.no_grad():
                while True:
                    val_action = self.agent.act(val_state)
                    next_val_state, val_reward, val_done, val_info = self.val_env.step(val_action)
                    val_loss = self.agent.calc_loss(val_state, val_action, next_val_state, val_reward, val_done)
                    val_state = next_val_state

                    self.logger.log_step(val_reward, val_loss, 'val_')
                    if val_done == 1:
                        self.logger.log_episode(val_info['acc'], val_info['f1'], 'val_')
                        break
            
            logs = self.logger.get_latest_logs()
            self.callbacks.on_epoch_end(episode, logs)
            if self.should_stop == True:
                break
        
        self.callbacks.on_validation_end()
        self.callbacks.on_train_end()
        LOG.info('Training finished')