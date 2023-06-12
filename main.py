""" main.py """

from configs.config import CFG
from utils.config import Config
from data_loaders.vn30_loader import DataLoader
from environments.vn30_trend_env import VN30TrendEnv
from executors.ppo_trainer import *
from loggers.logger import *
from gymnasium.wrappers import FrameStack
import gymnasium
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def env_factory(data_loader, env_dim, action_dim, env_cfg, seed, ticket, mode, timesteps_dim):
    return lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, env_cfg, seed, ticket, mode), num_stack=timesteps_dim)
    #return lambda: print(f'ticket: {ticket}')


def run():
    seed = 13
    set_seed(seed)
    config = Config.from_json(CFG)
    data_loader = DataLoader.from_json(config.data)
    tickets = data_loader.get_tickets()
    trading_days = data_loader.get_trading_days()
    timesteps_dim = data_loader.get_timesteps_dim()
    features_dim = data_loader.get_features_dim()
    
    env_dim = features_dim
    state_dim = timesteps_dim * features_dim
    action_dim = 1
    info_keys = ['true', 'prediction']  

    env_fns = [env_factory(data_loader, env_dim, action_dim, config.agent.env, seed, ticket, 'train', timesteps_dim) for ticket in tickets] #
    #print(len([env_fn().ticket  for env_fn in env_fns]))
    
    """ env_fns = [
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='ACB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BCM', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BID', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BVH', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='CTG', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='FPT', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='GAS', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='GVR', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='HDB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='HPG', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MBB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MSN', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MWG', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='NVL', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='PDR', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='PLX', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='POW', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='SAB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='SSI', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='STB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='TCB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='TPB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VCB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VHM', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VIB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VIC', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VJC', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VNM', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VPB', mode='train'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VRE', mode='train'), num_stack=timesteps_dim),
    ] """
    #[print(env_fn().ticket)  for env_fn in env_fns]
    env = gymnasium.vector.SyncVectorEnv(env_fns)
    
    test_env_fns = [env_factory(data_loader, env_dim, action_dim, config.agent.env, seed, ticket, 'test', timesteps_dim) for ticket in tickets]
    """ test_env_fns = [
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='ACB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BCM', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BID', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='BVH', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='CTG', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='FPT', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='GAS', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='GVR', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='HDB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='HPG', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MBB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MSN', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='MWG', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='NVL', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='PDR', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='PLX', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='POW', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='SAB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='SSI', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='STB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='TCB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='TPB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VCB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VHM', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VIB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VIC', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VJC', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VNM', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VPB', mode='test'), num_stack=timesteps_dim),
                lambda: FrameStack(VN30TrendEnv(data_loader, env_dim, action_dim, config.agent.env, seed, ticket='VRE', mode='test'), num_stack=timesteps_dim),
    ] """
    test_env = gymnasium.vector.SyncVectorEnv(test_env_fns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TrainLogger(info_keys, tickets, trading_days)
    test_logger = TestLogger(info_keys, tickets, trading_days)

    trainer = PPO(device, state_dim, action_dim, config.agent.train, config.agent.save, logger, test_logger)
    trainer.train(env, test_env, seed, config.agent.env)
    
    
    """ state, _ = env.reset(options={'mode' : 'episode', 'episode' : 100})
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
    print(truncated.shape)
    print(test_env) """
if __name__ == "__main__":
    run()
