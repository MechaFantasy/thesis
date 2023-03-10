# internal
from utils.logger import get_logger
LOG = get_logger('trainer')

# external
from environment.trend_env import TrendEnv


class TrendAgentTrainer:
    
    @staticmethod
    def train(agent, data_loader, logger, env_mode):
        LOG.info('Training started')
        x_train, y_train = data_loader.get_3D_train()
        env = TrendEnv(x_train, y_train, env_mode)
        
        for e in range(agent.episodes):

            state = env.reset()
            while True:
                action = agent.act(state)
                # Agent performs action
                next_state, reward, done, info = env.step(action)
                # Remember
                agent.cache(state, next_state, action, reward, done)
                # Learn
                loss = agent.learn()
                # Logging
                logger.log_step(reward, loss)
                # Update state
                state = next_state
                # Check if end of game
                if done == 1:
                    logger.log_episode(info['acc'], info['f1'])
                    break

            if e % 20 == 0:
                logger.record(episode=e)
        LOG.info('Training finished')