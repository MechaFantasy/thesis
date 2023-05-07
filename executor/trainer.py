# internal
from utils.logger import get_logger
from ops.support_func import *
from agent.agent import *
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
        

class A2CTrainer:
    
    def __init__(self, agent, device, env, test_env, logger, test_logger, train_cfg, save_cfg, env_cfg):
        self.gamma = train_cfg.gamma
        self.batch_size = train_cfg.batch_size
        self.lr = train_cfg.lr
        self.entropy_coeff = train_cfg.entropy_coeff
        self.episodes = train_cfg.episodes
        self.memory = ReplayBuffer()

        self.device = device
        self.agent = agent
        self.agent.to(device)
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=self.lr)
        self.env = env
        self.test_env = test_env
        self.logger = logger
        self.test_logger = test_logger

        self.save_specs = f'rconst{env_cfg.reward_const}-thres{env_cfg.transaction_thres}/' +\
                             f'g{self.gamma}-b{self.batch_size}-lr{self.lr}-ec{self.entropy_coeff}/'
        self.chkpts_path = save_cfg.base + save_cfg.checkpoints + self.save_specs
        self.logs_path = save_cfg.base + save_cfg.logs + self.save_specs
        self.save_freq = save_cfg.save_freq

        self.ticket_random_lst = []

    def __load(self):
        chkpt_files = os.listdir(self.chkpts_path)
        latest_chkpt_file = 'latest.chkpt'
        best_chkpt_file = 'best.chkpt' 
        if (latest_chkpt_file in chkpt_files) and (best_chkpt_file in chkpt_files):
            latest_chkpt_dict = torch.load(self.chkpts_path + latest_chkpt_file)
            best_chkpt_dict = torch.load(self.chkpts_path + best_chkpt_file)

            self.global_step = latest_chkpt_dict['step'] + 1
            self.start_episode = latest_chkpt_dict['episode'] + 1
            self.end_episode = self.start_episode + self.episodes
            self.latest_accuracy = latest_chkpt_dict['accuracy']
            self.agent.policy.load_state_dict(latest_chkpt_dict['policy'])
            self.optimizer.load_state_dict(latest_chkpt_dict['optimizer'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            self.best_accuracy = best_chkpt_dict['accuracy']
            self.best_policy_w = best_chkpt_dict['policy']
            self.best_optimizer_w = best_chkpt_dict['optimizer']
        else:
            self.global_step = 0
            self.start_episode = 0
            self.end_episode = self.episodes
            self.latest_accuracy = 0.3

            self.best_accuracy = 0.3
            self.best_policy_w = self.agent.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()
        
        """ print(self.global_step)
        print(self.start_episode)
        print(self.end_episode)
        print(self.latest_accuracy)
        print(self.best_accuracy)
        exit() """

    def __init_train(self):
        os.makedirs(self.chkpts_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        self.agent.to(self.device)
        self.writer = SummaryWriter(self.logs_path)
        self.__load()

    def __train_one_episode(self, episode):
        LOG.info(f'Training phase: -------------------------')
        self.logger.init_episode()
        step = 0
        state, _ = self.env.reset() #options={'mode' : 'episode', 'episode' : episode}
        
        #self.ticket_random_lst.append(self.env.ticket)

        action_lst, action_mean_lst, action_std_lst, value_lst, reward_lst = [], [], [], [], []
        while True:
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray
            action, log_prob, entropy, value, action_mean, action_std = self.agent.act_train(ToTorch(state, self.device))
            next_state, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())

            action_lst.append(action)
            action_mean_lst.append(action_mean)
            action_std_lst.append(action_std)
            value_lst.append(value)
            reward_lst.append(reward)
            self.logger.log_step(info)
            self.memory.append({
                'log_prob': log_prob, 'entropy': entropy, 'value': value,\
                'reward': ToTorch([reward], device=self.device),\
                'terminated': ToTorch([terminated], device=self.device)
            })

            self.writer.add_scalar('Train/action', action, self.global_step)
            self.writer.add_scalar('Train/log_prob', log_prob, self.global_step)
            self.writer.add_scalar('Train/value', value, self.global_step)
            self.writer.add_scalar('Train/action_mean', action_mean, self.global_step)
            self.writer.add_scalar('Train/action_std', action_std, self.global_step)
            self.writer.add_scalar('Train/reward', reward, self.global_step)        

            if terminated or ((step + 1) % self.batch_size == 0):
                items = self.memory.zip()
                log_probs, entropies, values, rewards, terminateds = items['log_prob'], items['entropy'], items['value'], items['reward'], items['terminated']

                _, _, _, next_value, _, _ = self.agent.act_train(ToTorch(next_state, self.device))
                returns = compute_returns(next_value.detach(), rewards, terminateds, self.gamma)
                assert returns.requires_grad == False
                assert (returns.shape == values.shape) and (returns.shape == entropies.shape) and (log_probs.shape == values.shape)

                advantage = returns - values
                actor_loss  = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()
                entropy_loss = self.entropy_coeff * entropies.mean()

                loss = actor_loss + critic_loss - entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                LOG.info(f'Step: {step}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy_loss {entropy_loss}')

                self.writer.add_scalar('Train/advantage', advantage.detach().mean(), self.global_step)
                self.writer.add_scalar('Train/actor_loss', actor_loss, self.global_step)
                self.writer.add_scalar('Train/critic_loss', critic_loss, self.global_step)
                self.writer.add_scalar('Train/entropy_loss', entropy_loss, self.global_step)
                self.writer.flush()
                self.memory.clear()
            
            state = next_state
            step += 1
            self.global_step += 1

            if terminated:  break

        accuracy, f1, cf_mt_dict = self.logger.record()
        LOG.info(f'Global step: {self.global_step - 1}, Accuracy: {accuracy}, F1: {f1}')
        self.writer.add_scalar('Train/Accuracy', accuracy, episode)
        self.writer.add_scalar('Train/F1', f1, episode)
        self.writer.add_scalars('Train/Normalized_confusion_matrix', cf_mt_dict, episode)

        self.writer.add_histogram('Train/Action', ToTorch(action_lst, self.device), episode)
        self.writer.add_histogram('Train/Action_mean', ToTorch(action_mean_lst, self.device), episode)
        self.writer.add_histogram('Train/Action_std', ToTorch(action_std_lst, self.device), episode)
        self.writer.add_histogram('Train/Reward_histogram', ToTorch(reward_lst, self.device), episode)
        self.writer.add_histogram('Train/Value_histogram', ToTorch(value_lst, self.device), episode)
        self.writer.flush()
        
        if episode % self.save_freq == 0:
            self.latest_accuracy = accuracy
            torch.save( dict(episode=episode, step=(self.global_step -1), policy=self.agent.policy.state_dict(),\
                            optimizer=self.optimizer.state_dict(), accuracy=accuracy),
                        self.chkpts_path + f'{episode}.chkpt')
        LOG.info(f'Save episode {episode}')


    def __test_one_episode(self, episode):
        LOG.info(f'Testing phase: -------------------------')
        self.test_logger.init_episode()
        step = 0
        state, _ = self.test_env.reset()
        action_lst = []
        while True:
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray
            action = self.agent.act_test(ToTorch(state, self.device))
            next_state, reward, terminated, truncated, info = self.test_env.step(action.cpu().numpy())

            self.test_logger.log_step(info, terminated)
            action_lst.append(action)
            state = next_state
            step += 1
            if terminated.any():  break
        
        self.writer.add_histogram('Test/Action', torch.cat(action_lst).to(self.device), episode)
            

        accuracy, f1, cf_mt_dict = self.test_logger.record(episode)
        LOG.info(f'Accuracy: {accuracy}, F1: {f1}')
        self.writer.add_scalar('Test/Accuracy', accuracy, episode)
        self.writer.add_scalar('Test/F1', f1, episode)
        self.writer.add_scalars('Test/Normalized_confusion_matrix', cf_mt_dict, episode)
        self.writer.flush()
        if accuracy >= self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_policy_w = self.agent.policy.state_dict()
            self.best_optimizer_w = self.optimizer.state_dict()

    def train(self):
        source_dir = './configs/' 
        destination_dir = './configs/' + self.save_specs
        file_name = 'config.py'
        if os.path.exists(destination_dir) == False:
            os.makedirs(destination_dir, exist_ok=True)
        shutil.copyfile(source_dir + file_name, destination_dir + file_name)
        
        LOG.info('Training started')
        self.__init_train()
    
        for episode in range(self.start_episode, self.end_episode):
            LOG.info('======================================================================================================')
            LOG.info(f'Episode {episode}/{self.end_episode - 1}')
            self.__train_one_episode(episode)
            self.__test_one_episode(episode)

        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.best_policy_w,\
                        optimizer=self.best_optimizer_w, accuracy=self.best_accuracy),
                    self.chkpts_path + 'best.chkpt')
        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.agent.policy.state_dict(),\
                        optimizer=self.optimizer.state_dict(), accuracy=self.latest_accuracy),
                    self.chkpts_path + 'latest.chkpt')
        self.writer.close()

        ###################################################
        """ ticket_dir_path = './debug/ticket_random_list/' + self.save_specs
        ticket_file_path = ticket_dir_path + f'ticket.csv'
        os.makedirs(ticket_dir_path, exist_ok=True)
        if os.path.exists(ticket_file_path) == False:
            ticket_df = pd.DataFrame(self.ticket_random_lst, columns=['ticket'])
            ticket_df.index.name = 'episode'
        else:
            exist_ticket_df = pd.read_csv(ticket_file_path, index_col='episode')
            ticket_df = pd.DataFrame(self.ticket_random_lst, columns=['ticket'])
            ticket_df.index.name = 'episode'
            ticket_df = pd.concat([exist_ticket_df, ticket_df], axis=0, ignore_index=True)
        ticket_df.to_csv(ticket_file_path) """
        ###################################################
        
        LOG.info(f'Config specifics: {self.save_specs}')
        LOG.info('Training finished')


        
        

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
        self.logs_path = self.base + save_cfg.logs + self.train_specs
        self.cfgs_path = self.base + save_cfg.configs + self.train_specs
        self.figs_path = self.base + save_cfg.figs + self.train_specs
        self.save_freq = 30

        self.logger = logger
        self.test_logger = test_logger

    def __load(self):
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
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()
    
    def select_action_test(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_old.act_test(state)
        
        return action.detach().cpu().numpy()


    def update(self, next_state, terminated):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        if terminated:
          discounted_reward = 0
        else:
          with torch.no_grad():
            next_state = torch.FloatTensor(next_state).to(self.device)
            discounted_reward = self.policy_old.critic(next_state).item()
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

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def __init_train(self, env_cfg=''):
        env_specs = f'rconst{env_cfg.reward_const}-thres{env_cfg.transaction_thres}/'
        self.chkpts_path += env_specs 
        self.logs_path += env_specs
        self.cfgs_path += env_specs
        os.makedirs(self.chkpts_path , exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.cfgs_path, exist_ok=True)

        filename = 'config.py'
        shutil.copyfile('./configs/' + filename, self.cfgs_path + filename)
        self.writer = SummaryWriter(self.logs_path)
        self.__load()

    def __train_one_episode(self, env, episode, seed):
        state, _ = env.reset(seed=seed) 
        current_ep_reward = 0
        step = 0
        self.logger.init_episode()

        while True:
            state = state if type(state).__module__ == np.__name__ else state.__array__() #return ndarray

            action = self.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            self.logger.log_step(info)
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(terminated)

            self.writer.add_scalar('Train/action', action, self.global_step)
            self.writer.add_scalar('Train/step_reward', reward, self.global_step) 
            self.writer.flush()      
            if terminated or ((step + 1) % self.batch_size == 0) :
                self.update(next_state, terminated)

            step += 1
            self.global_step += 1
            current_ep_reward += reward
            state = next_state

            if terminated:  break
        
        accuracy, f1, cf_mt_dict = self.logger.record(episode)
        LOG.info(f'Global step: {self.global_step - 1}, Accuracy: {accuracy}, F1: {f1}')
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
            self.writer.add_scalar('Test/step_reward', reward, self.global_step) 

            step += 1
            current_ep_reward += reward
            state = next_state
            if terminated:  break
        
        accuracy, f1, cf_mt_dict = self.test_logger.record(episode)
        LOG.info(f'Accuracy: {accuracy}, F1: {f1}')
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
        self.__init_train(env_cfg)
        for episode in range(self.start_episode, self.end_episode):
            print(f'Episode {episode}/{self.end_episode - 1}')
            self.__train_one_episode(env, episode, seed)
            self.__test_one_episode(test_env, episode, seed )

            if ((episode + 1) % self.action_std_decay_freq == 0):
                self.decay_action_std()

        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.best_policy_w,\
                        optimizer=self.best_optimizer_w, reward=self.best_reward, action_std=self.action_std),
                    self.chkpts_path + 'best.chkpt')
        torch.save( dict(episode=(self.end_episode - 1), step=(self.global_step - 1), policy=self.policy.state_dict(),\
                        optimizer=self.optimizer.state_dict(), reward=self.latest_reward, action_std=self.action_std),
                    self.chkpts_path + 'latest.chkpt')
        self.writer.close()
        LOG.info(f'Config specifics: {self.train_specs}')
        LOG.info('Training finished')