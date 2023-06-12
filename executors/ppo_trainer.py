# internal
from utils.logger import get_logger
from ops.support_func import *
from agents.agent import *
# external
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import torch.optim as optim
import shutil
import pandas as pd
torch.autograd.set_detect_anomaly(True)
LOG = get_logger('Trainer')

def ToTorch(obj, device='cpu'):
    return torch.tensor(obj, dtype=torch.float).to(device)

def compute_returns(next_value, rewards, terminated, gamma):
    R = next_value
    returns = rewards.clone()
    for ind in reversed(range(rewards.shape[0])):
        R = rewards[ind] + gamma * R * (1 - terminated[ind])
        returns[ind] = R.item()
    return returns
        
class PPO:
    
    def __init__(self, device, state_dim, action_dim, train_cfg, save_cfg, logger, test_logger):
        self.episodes = train_cfg.episodes

        ##hyperparameters
        self.critic_coeff = train_cfg.critic_coeff
        self.entropy_coeff = train_cfg.entropy_coeff
        self.eps_clip = train_cfg.eps_clip             # clip parameter for PPO
        self.gamma = train_cfg.gamma                # discount factor
        self.batch_size = train_cfg.batch_size #max_ep_len * 4      # update policy every n timesteps
        self.lr_actor = train_cfg.lr_actor    # learning rate for actor network
        self.lr_critic = train_cfg.lr_critic       # learning rate for critic network
        self.K_epochs = train_cfg.K_epochs              # update policy for K epochs
        self.action_std = train_cfg.action_std                  # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = train_cfg.action_std_decay_rate       # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = train_cfg.min_action_std               # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = train_cfg.action_std_decay_freq 
        
        self.device = device
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, self.action_std, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.action_std, self.device).to(self.device)
        self.MseLoss = nn.MSELoss()

        self.train_specs = f'ccoeff{self.critic_coeff}-ecoeff{self.entropy_coeff}-eps_clip{self.eps_clip}-g{self.gamma}-b{self.batch_size}-lr_actor{self.lr_actor}-lr_critic{self.lr_critic}-K_eps{self.K_epochs}-a_std{self.action_std}-a_std_drate{self.action_std_decay_rate}-min_a_std{self.min_action_std}-a_std_dfreq{self.action_std_decay_freq}/'
        self.base = save_cfg.base
        self.chkpts_path = self.base + save_cfg.checkpoints + self.train_specs
        self.tensorboards_path = self.base + save_cfg.tensorboards + self.train_specs
        self.cfgs_path = self.base + save_cfg.configs + self.train_specs
        self.figs_path = self.base + save_cfg.figs + self.train_specs
        self.loggers_path = self.base + save_cfg.loggers + self.train_specs
        self.save_freq = 30

        self.logger = logger
        self.test_logger = test_logger

    def __load_chkpt(self):
        chkpt_files = os.listdir(self.chkpts_path)
        latest_chkpt_file = 'latest.chkpt'
        best_chkpt_file = 'best.chkpt' 
        if (latest_chkpt_file in chkpt_files) and (best_chkpt_file in chkpt_files):
            latest_chkpt_dict = torch.load(self.chkpts_path + latest_chkpt_file)
            best_chkpt_dict = torch.load(self.chkpts_path + best_chkpt_file)

            self.global_step = latest_chkpt_dict['step'] + 1
            self.start_episode = latest_chkpt_dict['episode'] + 1
            self.end_episode = self.start_episode + self.episodes
            self.latest_reward = latest_chkpt_dict['reward']

            self.action_std = latest_chkpt_dict['action_std']
            self.policy.load_state_dict(latest_chkpt_dict['policy'])
            self.optimizer.load_state_dict(latest_chkpt_dict['optimizer'])
            self.set_action_std(self.action_std)
            """ for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr """

            self.best_reward = best_chkpt_dict['reward']
            self.best_policy_w = best_chkpt_dict['policy']
            self.best_optimizer_w = best_chkpt_dict['optimizer']
        else:
            self.global_step = 0
            self.start_episode = 0
            self.end_episode = self.episodes
            self.latest_reward = np.NINF

            self.best_reward = np.NINF
            self.best_policy_w = self.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()   
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __init_train(self, env_cfg):
        env_specs = f'rconst{env_cfg.reward_const}-thres{env_cfg.transaction_thres}/' 
        self.chkpts_path += env_specs 
        self.tensorboards_path += env_specs
        self.cfgs_path += env_specs
        self.figs_path += env_specs
        self.loggers_path += env_specs
        os.makedirs(self.chkpts_path , exist_ok=True)
        os.makedirs(self.tensorboards_path, exist_ok=True)
        os.makedirs(self.cfgs_path, exist_ok=True)
        os.makedirs(self.figs_path, exist_ok=True)
        os.makedirs(self.loggers_path, exist_ok=True)

        filename = 'config.py'
        shutil.copyfile('./configs/' + filename, self.cfgs_path + filename)
        self.writer = SummaryWriter(self.tensorboards_path)
        self.__load_chkpt()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
    def decay_action_std(self):
        print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= self.min_action_std):
            self.action_std = self.min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val, action_mean, action_std = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        """ print(f'action {action.shape}, action_mean {action_mean.shape}, action_std {action_std.shape}, state_val {state_val.shape}')
        exit() """

        self.writer.add_scalar('Train/action', action.mean(), self.global_step)
        self.writer.add_scalar('Train/action_mean', action_mean.mean(), self.global_step)
        self.writer.add_scalar('Train/action_std', action_std.mean(), self.global_step)  
        self.writer.add_scalar('Train/state_val', state_val.mean(), self.global_step)  

        return action.detach().cpu().numpy() #.flatten()
    
    def select_action_test(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_old.act_test(state)
        
        return action.detach().cpu().numpy()

    def update(self, next_state, terminated):
        # Monte Carlo estimate of returns
        n_env = next_state.shape[0]
        rewards = []
        if terminated.any():
            discounted_reward = np.zeros(n_env, dtype=np.float32)
        else:
          with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device)
            discounted_reward = self.policy_old.critic(next_state).flatten().cpu().numpy()
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward) 
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = self.critic_coeff * self.MseLoss(state_values, rewards)
            entropy_loss = (- self.entropy_coeff * dist_entropy).mean()
            loss = actor_loss + critic_loss + entropy_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if k == (self.K_epochs - 1):
                self.writer.add_scalar('Train/actor_loss', actor_loss.item(), self.global_step)
                self.writer.add_scalar('Train/critic_loss', critic_loss.item(), self.global_step)
                self.writer.add_scalar('Train/entropy_loss', entropy_loss.item(), self.global_step)
                self.writer.flush()  
                LOG.info(f'Global step: {self.global_step}, Buffer length: {self.buffer.len()}, Actor loss: {actor_loss.item()}, Critic loss: {actor_loss.item()}, Entropy loss: {entropy_loss.item()}')

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def __train_one_episode(self, env, episode, seed):
        state, _ = env.reset(seed=seed) 
        current_ep_reward = 0
        step = 0
        self.logger.init_episode()

        while True:
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return state ndarray

            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(terminated)
            
            self.logger.log_step(info, terminated)
            self.writer.add_scalar('Train/step_reward', np.mean(reward), self.global_step) 
            self.writer.flush()      
           
            if terminated.any() or ((step + 1) % self.batch_size == 0) :
                self.update(next_state, terminated)

            step += 1
            self.global_step += 1
            current_ep_reward += np.mean(reward)
            state = next_state

            if terminated.any():  break
        
        accuracy, f1, cf_mt_dict = self.logger.record(episode)
        LOG.info(f'Train tickets: {env.call("ticket")}')
        LOG.info(f'Train reward: {current_ep_reward}, Train Accuracy: {accuracy}, Train F1: {f1}')
        self.writer.add_scalar('Train/Accuracy', accuracy, episode)
        self.writer.add_scalar('Train/F1', f1, episode)
        self.writer.add_scalars('Train/Normalized_confusion_matrix', cf_mt_dict, episode)
        self.writer.add_scalar('Train/reward', current_ep_reward, episode)
        self.writer.flush()
        self.latest_reward = current_ep_reward
        if episode % self.save_freq == 0:
            torch.save( dict(episode=episode, step=(self.global_step -1), policy=self.policy_old.state_dict(),\
                            optimizer=self.optimizer.state_dict(), reward=current_ep_reward, action_std=self.action_std),
                        self.chkpts_path + f'{episode}.chkpt')
            LOG.info('++++++++++++++++++++++')
            LOG.info(f'Saving checkpoint at: {self.chkpts_path}{episode}.chkpt')
            LOG.info('++++++++++++++++++++++')


    def __test_one_episode(self, test_env, episode, seed):
        state, _ = test_env.reset(seed=seed)
        current_ep_reward = 0
        step = 0
        self.test_logger.init_episode()
        while True:
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray

            action = self.select_action_test(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            self.test_logger.log_step(info, terminated)
            
            step += 1
            current_ep_reward += np.mean(reward)
            state = next_state
            if terminated.any():  break
        
        accuracy, f1, cf_mt_dict = self.test_logger.record(episode)
        LOG.info(f'Test tickets: {test_env.call("ticket")}')
        LOG.info(f'Test reward: {current_ep_reward}, Test Accuracy: {accuracy}, Test F1: {f1}')
        self.writer.add_scalar('Test/Accuracy', accuracy, episode)
        self.writer.add_scalar('Test/F1', f1, episode)
        self.writer.add_scalars('Test/Normalized_confusion_matrix', cf_mt_dict, episode)
        self.writer.add_scalar('Test/reward', current_ep_reward, episode)  
        self.writer.flush()
        if current_ep_reward >= self.best_reward:
            self.best_reward = current_ep_reward
            self.best_policy_w = self.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()

    def train(self, env, test_env, seed, env_cfg):
        LOG.info('Training started')
        LOG.info('=======================================================================================================')
        self.__init_train(env_cfg)
        for episode in range(self.start_episode, self.end_episode):
            LOG.info(f'-----------------------------Episode {episode}/{self.end_episode - 1}-----------------------------')
            self.__train_one_episode(env, episode, seed)
            self.__test_one_episode(test_env, episode, seed)

            if ((episode + 1) % self.action_std_decay_freq == 0):
                self.decay_action_std()

        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.best_policy_w,\
                        optimizer=self.best_optimizer_w, reward=self.best_reward, action_std=self.action_std),
                    self.chkpts_path + 'best.chkpt')
        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.policy.state_dict(),\
                        optimizer=self.optimizer.state_dict(), reward=self.latest_reward, action_std=self.action_std),
                    self.chkpts_path + 'latest.chkpt')

        LOG.info('=======================================================================================================')
        self.writer.close()
        LOG.info(f'Tensorboard log path: {self.tensorboards_path}')
        LOG.info(f'Close tensorboard!\n')

        max_train_idx, max_train_accuracy, max_train_f1 = self.logger.to_csv(self.loggers_path)
        LOG.info(f'Episode: {max_train_idx}, max train accuracy: {max_train_accuracy}, max train f1: {max_train_f1}')
        max_test_idx, max_test_accuracy, max_test_f1 = self.test_logger.to_csv(self.loggers_path)
        LOG.info(f'Episode: {max_test_idx}, max test accuracy: {max_test_accuracy}, max test f1: {max_test_f1}\n')

        LOG.info('Training finished')