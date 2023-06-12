import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VN30TrendEnv(gym.Env):

    
    def __init__(self, data_loader, env_dim, action_dim, env_cfg, seed, ticket='ACB', mode='train'): 
        self.vn30_dict = data_loader.get_vn30_df_dict()
        self.vn30_tickets = data_loader.get_tickets()
        self.trading_days = data_loader.get_trading_days()
        self.n_predictions = data_loader.get_n_predictions()
        self.seed = seed
        self.ticket = ticket
        self.mode = mode
        

        train_start_day_ind, train_end_day_ind, test_start_day_ind, test_end_day_ind = data_loader.get_days_ind_train_test_split(data_loader.get_test_day_ind())
        if self.mode == 'train':
            self.start_day_ind, self.end_day_ind = train_start_day_ind, train_end_day_ind
        else:
            self.start_day_ind, self.end_day_ind = test_start_day_ind, test_end_day_ind

        self.floor_price_change, self.ceil_price_change = -0.07, 0.07
        self.transaction_thres = env_cfg.transaction_thres
        self.reward_const = env_cfg.reward_const
        self.info_keys = ['true', 'prediction']
        self.flat_trend_thres = self.transaction_thres / self.ceil_price_change

        self.action_dim = action_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim, ), dtype='float32')
        self.env_dim = env_dim
        self.obs_shape = (self.env_dim, )
        self.observation_space = spaces.Box(low=np.NINF, high=np.PINF, shape=(self.env_dim, ), dtype='float32')
    
    def get_info_keys(self):
        return self.info_keys

    def __get_state(self):
        state_days = self.trading_days[self.curr_day_ind - self.n_predictions]
        state_df = self.vn30_dict[self.ticket].loc[state_days, :]
        state_np = np.reshape(state_df.to_numpy(dtype='float32'), self.obs_shape)
        return state_np
    
    def __get_reward_and_info(self, action):
        action = action.squeeze()
        curr_day_price_change = ((self.vn30_dict[self.ticket].loc[self.trading_days[self.curr_day_ind], 'close']) - 1).squeeze() #scalar
        reward2 = (self.reward_const - np.abs(curr_day_price_change * 100 - action * 7) / 14) * 20
        info = {}
        
        """ info['true'] = np.where(curr_day_price_change > self.transaction_thres, 2,\
                        np.where(curr_day_price_change < -self.transaction_thres, 0, 1)) #scalar
        info['prediction'] = np.where(action > self.flat_trend_thres, 2,\
                                np.where(action < -self.flat_trend_thres, 0, 1)) #scalar   """
        info['true'] = np.where(curr_day_price_change > 0, 1, 0) 
        info['prediction'] = np.where(action > 0, 1, 0) #scalar  
        return reward2, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options:
            if options['mode'] == 'episode':
                episode = options['episode']
                ticket_ind = episode % len(self.vn30_tickets)
                self.ticket = self.vn30_tickets[ticket_ind]

        self.curr_day_ind = self.n_predictions if self.start_day_ind < self.n_predictions else self.start_day_ind
        state = self.__get_state()
        info = {}
        return state, info
    
    def step(self, action):
        assert action.shape[0] == self.action_dim, f'action_shape {action.shape}, action_dim {self.action_dim}'
        assert type(action).__module__ == np.__name__

        action = np.clip(action, -1, 1)

        reward, info = self.__get_reward_and_info(action)

        terminated = 0
        if self.curr_day_ind == self.end_day_ind:
            terminated = 1
            next_state, _ = self.reset()
        else:
            self.curr_day_ind += 1
            next_state = self.__get_state()

        return next_state, reward, terminated, False, info
    