""" main.py """

from configs.config import CFG
from agent.trend_agent import TrendAgent
from utils.config import Config
from dataloader.dataloader import DataLoader

def run():
    config = Config.from_json(CFG)
    data_loader = DataLoader.from_json(config.data)
    agent = TrendAgent(config.agent, data_loader.get_input_dim())
    #agent.train(data_loader)
    agent.predict(data_loader)


if __name__ == "__main__":
    run()
