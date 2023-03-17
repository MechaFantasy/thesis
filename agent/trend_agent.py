# -*- coding: utf-8 -*-
"""Unet model"""

# standard library
import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import os
# external
from .base_agent import BaseAgent
from utils.logger import get_logger
from executor.trend_agent_trainer import TrendAgentTrainer
from .model import *
from .metrics_logger import MetricsLogger
from environment.trend_env import TrendEnv
from callbacks.callbacks import *
# internal
LOG = get_logger('trend_agent')

class TrendAgent(BaseAgent):
    
    
    def __init__(self, cfg, input_dim):
        super().__init__(cfg)
        self.state_dim = input_dim
        seq_len = input_dim[0]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = cfg.train.model_type
        if self.model_type == 'CNN':
            self.policy_net = CNNTrendNet(self.state_dim, cfg.model.CNN).to(device=self.device)
            self.target_net = CNNTrendNet(self.state_dim, cfg.model.CNN).to(device=self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        elif self.model_type == 'LSTM':
            self.policy_net = LSTMTrendNet(self.state_dim, cfg.model.LSTM).to(device=self.device)
            self.target_net = LSTMTrendNet(self.state_dim, cfg.model.LSTM).to(device=self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
            
        self.action_dim = 2

        self.exploration_rate = cfg.train.exploration_rate
        self.exploration_rate_min = cfg.train.exploration_rate_min
        self.exploration_decay = cfg.train.exploration_decay
        self.curr_step = 0
        self.gamma = cfg.train.gamma

        self.replay_memory = deque(maxlen=cfg.train.replay_memory_size)
        self.batch_size = cfg.train.batch_size
        self.episodes = cfg.train.episodes
        self.start_from_episode = cfg.train.start_from_episode

        self.sync_every = cfg.train.sync_every
        self.learn_every = cfg.train.learn_every
        self.burnin = cfg.train.burnin

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.train.lr)
        self.patience = cfg.train.patience
        self.env_mode = cfg.env.mode

        
        save_specs = f'{self.env_mode}/' + f'{self.model_type}-seq{seq_len}-g{self.gamma}-m{cfg.train.replay_memory_size}-b{self.batch_size}-e{self.episodes}-sync{self.sync_every}-le{self.learn_every}-bu{self.burnin}-p{self.patience}/'
        self.checkpoints = cfg.save.base + cfg.save.checkpoints + save_specs
        self.logs = cfg.save.base + cfg.save.logs + save_specs
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        if np.random.rand() < self.exploration_rate:
            trend_idx = np.random.randint(self.action_dim)
        
        else:
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            trend_values = self.policy_net(state)
            trend_idx = torch.argmax(trend_values, axis=1).item()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)
        self.curr_step += 1
        return trend_idx
    
    def td_estimate(self, state, action):
        current_Q = self.policy_net(state)[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.policy_net(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_net(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def sync_Q_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.replay_memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.replay_memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return 0

        if self.curr_step % self.learn_every != 0:
            return 0

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calc_loss(self, state, action, next_state, reward, done):
        torch_state = torch.tensor(state, device=self.device).unsqueeze(0)
        torch_next_state = torch.tensor(next_state, device=self.device).unsqueeze(0)
        torch_action = torch.tensor([action], device=self.device)
        torch_reward = torch.tensor([reward], device=self.device)
        torch_done = torch.tensor([done], device=self.device)
        
        td_est = self.policy_net(torch_state).squeeze(0)[torch_action]
        
        next_state_Q = self.policy_net(torch_next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.target_net(torch_next_state).squeeze(0)[best_action]
        td_tgt = (torch_reward + (1 - torch_done.float()) * self.gamma * next_Q)

        loss = self.loss_fn(td_est, td_tgt).item()
        return loss
    
    def load(self, best_agent_chkpt_file='best.chkpt'):
        chkpt_files = os.listdir(self.checkpoints)
        if best_agent_chkpt_file in chkpt_files:
            filepath = self.checkpoints + best_agent_chkpt_file
            checkpoint_dict = torch.load(filepath)
            self.policy_net.load_state_dict(checkpoint_dict['policy'])
            self.target_net.load_state_dict(checkpoint_dict['target'])
            self.exploration_rate = checkpoint_dict['exploration_rate']
        
    def train(self, data_loader):
        self.load()
        x_train, y_train, x_val, y_val = None, None, None, None
        if self.model_type == 'CNN':
            x_train, y_train = data_loader.get_3D_train()
            x_val, y_val = data_loader.get_3D_val()
        else: 
            x_train, y_train = data_loader.get_train()
            x_val, y_val = data_loader.get_val()

        env = TrendEnv(x_train, y_train, self.env_mode)
        val_env = TrendEnv(x_val, y_val, self.env_mode) 
        metrics = ['loss', 'reward', 'acc', 'f1']
        logger = MetricsLogger()
        callbacks = CallBackList(
                        [
                            AgentChechPoint(filepath=self.checkpoints, monitor='val_loss'), \
                            TensorBoard(log_dir=self.logs), \
                            EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience)
                        ]
                    )
        
        trainer = TrendAgentTrainer(self, env, val_env, metrics, logger, callbacks)  
        trainer.train()

                
    def predict(self, data_loader):
        LOG.info("Prediction started")
        self.load()
        x_test, y_test = None, None
        if self.model_type == 'CNN':
            x_test, y_test = data_loader.get_3D_test()
        else: 
            x_test, y_test = data_loader.get_test()
        env = TrendEnv(x_test, y_test, self.env_mode)
        total_reward = 0
        avg_loss = 0
        state = env.reset()
        with torch.no_grad():
            while True:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)

                avg_loss += self.calc_loss(state, action, next_state, reward, done)
                
                total_reward += reward 
                state = next_state
                if done == 1:
                    accuracy = info['acc']
                    f1 = info['f1']
                    print('---------------------')
                    print(f'Loss: {avg_loss / env.game_len}, Reward: {total_reward}, Accuracy: {accuracy}, F1-score: {f1}')
                    break
        
        LOG.info('Prediction finished')