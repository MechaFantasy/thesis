# internal
from utils.logger import get_logger
from dataloader.dataloader import DataLoader
LOG = get_logger('trainer')

# external
import os
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class DQNTrainer:
    
    
    def __init__(self, game_state, model, optimizer, criterion, train_cfg, image_size, saved_dir):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.game_state = game_state
        self.image_size = image_size
        self.replay_memory = []
        
        self.gamma = train_cfg.gamma
        self.final_epsilon = train_cfg.final_epsilon
        self.initial_epsilon = train_cfg.initial_epsilon
        self.number_of_iterations = train_cfg.number_of_iterations
        self.replay_memory_size = train_cfg.replay_memory_size
        self.minibatch_size = train_cfg.minibatch_size
        #self.lr = train_cfg.lr
        
        self.log_dir = saved_dir.logs
        self.checkpoints_dir = saved_dir.checkpoints
        self.writer = SummaryWriter(self.log_dir)
        
    def train(self):
        # initial action is do nothing
        start = time.time()
        action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
        action[0] = 1
        image_data, reward, terminal = self.game_state.frame_step(action)
        image_data = DataLoader().preprocess_data(image_data, self.image_size)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        # initialize epsilon value
        epsilon = self.initial_epsilon
        iteration = 0

        epsilon_decrements = np.linspace(self.initial_epsilon, self.final_epsilon, self.number_of_iterations)

        # main infinite loop
        while iteration < self.number_of_iterations:
            # get output from the neural network
            output = self.model(state)[0]

            # initialize action
            action = torch.zeros([self.model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action = action.cuda()

            # epsilon greedy exploration
            random_action = random.random() <= epsilon
            if random_action:
                print("Performed random action!")
            action_index = [torch.randint(self.model.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action
                            else torch.argmax(output)][0]

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                action_index = action_index.cuda()

            action[action_index] = 1

            # get next state and reward
            image_data_1, reward, terminal = self.game_state.frame_step(action)
            state_1 = DataLoader().produce_new_state(state, image_data_1, self.image_size)
            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            self.replay_memory.append((state, action, reward, state_1, terminal))

            # if replay memory is full, remove the oldest transition
            if len(self.replay_memory) > self.replay_memory_size:
                self.replay_memory.pop(0)

            # epsilon annealing
            epsilon = epsilon_decrements[iteration]

            # sample random minibatch
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # get output for the next state
            output_1_batch = self.model(state_1_batch)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                        else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                        for i in range(len(minibatch))))

            # extract Q-value
            q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            self.optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            loss = self.criterion(q_value, y_batch)

            # do backward pass
            loss.backward()
            self.optimizer.step()

            # set state to be state_1
            state = state_1
            iteration += 1

            if iteration % 50000 == 0:
                PATH = f"{self.checkpoints_dir}{iteration}.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss.item(),
                }, PATH)

            LOG.info(f"iteration: {iteration}, elapsed time: {time.time() - start}, epsilon: {epsilon}, action: {action_index.cpu().detach().numpy()}, reward: {reward.numpy()[0][0]},  Q max: {np.max(output.cpu().detach().numpy())}")
            self._write_summary(iteration, loss.item(), 'loss')
            
    def _write_summary(self, iteration, metrics, tag):
        self.writer.add_scalar(tag, metrics, iteration)
        self.writer.flush()
        
        # tensorboard --logdir logs