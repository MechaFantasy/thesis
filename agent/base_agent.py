from abc import ABC, abstractmethod
# -*- coding: utf-8 -*-
"""Abstract base model"""

from utils.config import Config


class BaseAgent(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.config = cfg

    @abstractmethod
    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass
    
    @abstractmethod
    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        pass

    @abstractmethod
    def recall(self):
        """Sample experiences from memory"""
        pass

    @abstractmethod
    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass