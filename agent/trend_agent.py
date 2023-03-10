# -*- coding: utf-8 -*-
"""Unet model"""

# standard library
import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import datetime
import os
# external
from .base_agent import BaseAgent
from utils.logger import get_logger
from executor.trend_agent_trainer import TrendAgentTrainer
from .model import TrendNet
from .metrics_logger import MetricsLogger
from environment.trend_env import TrendEnv
# internal
LOG = get_logger('trend_agent')

class TrendAgent(BaseAgent):
    
    
    def __init__(self, cfg, input_dim):
        super().__init__(cfg)
        self.state_dim = input_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_net = TrendNet(self.state_dim, cfg.model).to(device=self.device)
        self.target_net = TrendNet(self.state_dim, cfg.model).to(device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.action_dim = cfg.model.action_dim

        self.exploration_rate = cfg.train.exploration_rate
        self.exploration_rate_min = cfg.train.exploration_rate_min
        self.exploration_decay = cfg.train.exploration_decay
        self.curr_step = 0
        self.gamma = cfg.train.gamma

        self.relay_memory = deque(maxlen=cfg.train.relay_memory)
        self.batch_size = cfg.train.batch_size
        self.episodes = cfg.train.episodes

        self.sync_every = cfg.train.sync_every
        self.save_every = cfg.train.save_every
        self.learn_every = cfg.train.learn_every
        self.burnin = cfg.train.burnin

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.train.lr)
        
        self.env_mode = cfg.env.mode

        present_time = datetime.datetime.now()
        self.checkpoints = cfg.save.checkpoints + f'{self.env_mode}/{present_time}/' 
        self.logs = cfg.save.logs + f'{self.env_mode}/{present_time}/' 
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        
        self.load_path = self.checkpoints + f'trend_net_{cfg.test.load}.chkpt'

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
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.relay_memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.relay_memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def save(self):
        save_path = (
            self.checkpoints + f"trend_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(policy=self.policy_net.state_dict(), target=self.target_net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        LOG.info(f"Trend saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return 0

        if self.curr_step % self.learn_every != 0:
            return 0

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return loss
    
    def train(self, data_loader):      
        logger = MetricsLogger(self.logs)
        TrendAgentTrainer().train(self, data_loader, logger, self.env_mode)

    def load(self):
        checkpoint_dict = torch.load(self.load_path)
        self.policy_net.load_state_dict(checkpoint_dict['policy'])
        self.target_net.load_state_dict(checkpoint_dict['target'])
        self.exploration_rate = checkpoint_dict['exploration_rate']

                
    def evaluate(self, data_loader):
        LOG.info("Evaluation started")
        self.load()

        x_test, y_test = data_loader.get_3D_test()
        env = TrendEnv(x_test, y_test, self.env_mode)
        total_reward = 0
        state = env.reset()
        with torch.no_grad():
            while True:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)

                state = next_state
                total_reward += reward 
                if done == 1:
                    accuracy = info['acc']
                    f1 = info['f1']
                    print('---------------------')
                    print(f'Reward: {total_reward}, Accuracy: {accuracy}, F1-score: {f1}')
                    break
        
        LOG.info('Evaluation finished')
        