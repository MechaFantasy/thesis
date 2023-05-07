""" main.py """

from configs.config import CFG
from utils.config import Config
from dataloader.dataloader import DataLoader
from environment.env import VN30TrendEnv
from executor.trainer import *
from logger.logger import *
from gymnasium.wrappers import FrameStack
import gymnasium
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run():
    seed = 13
    set_seed(seed)
    config = Config.from_json(CFG)
    data_loader = DataLoader.from_json(config.data)
    vn30_tickets = data_loader.get_tickets()
    trading_days = data_loader.get_trading_days()
    timesteps_dim = data_loader.get_timesteps_dim()
    features_dim = data_loader.get_features_dim()
    
    env_dim = features_dim
    state_dim = timesteps_dim * features_dim
    action_dim = 1

    env = VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='ACB',mode='train') 
    env = FrameStack(env, num_stack=timesteps_dim)
    info_keys = env.get_info_keys()  
    env_fns = [
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='ACB', mode='test'), num_stack=timesteps_dim),
                
    ]
    """ lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='BCM', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='BID', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='BVH', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='CTG', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='FPT', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='GAS', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='GVR', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='HDB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='HPG', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='MBB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='MSN', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='MWG', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='NVL', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='PDR', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='PLX', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='POW', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='SAB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='SSI', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='STB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='TCB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='TPB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VCB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VHM', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VIB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VIC', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VJC', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VNM', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VPB', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, state_shape, action_shape, config.agent.env, seed, ticket='VRE', ticket_mode='deterministic', mode='test'), num_stack=timesteps_dim), """
    test_env = gymnasium.vector.SyncVectorEnv(env_fns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TrainLogger(info_keys, vn30_tickets, trading_days)
    test_logger = TestLogger(info_keys, vn30_tickets, trading_days)

    trainer = PPO(device, state_dim, action_dim, config.agent.train, config.agent.save, logger, test_logger)
    trainer.train(env, test_env, seed, config.agent.env)
    """ print(vn30_tickets)
    state, _ = env.reset(options={'mode' : 'episode', 'episode' : 100})
    print(state.__array__())
    next_state, reward, terminated, truncated, info = env.step(np.array([-1]))
    print(next_state.__array__())
    print(reward)
    print(terminated)
    print(info)
    print(truncated)
    
    print(test_env.observation_space)
    print(test_env.action_space)

    state, info = test_env.reset()
    print(state.shape)
    print(info)
    next_state, reward, terminated, truncated, info = test_env.step(np.ones((30, 1), dtype='float32'))
    print(next_state.shape)
    print(reward.shape)
    print(terminated.shape)
    print(info)
    print(truncated.shape) """
    
if __name__ == "__main__":
    run()
