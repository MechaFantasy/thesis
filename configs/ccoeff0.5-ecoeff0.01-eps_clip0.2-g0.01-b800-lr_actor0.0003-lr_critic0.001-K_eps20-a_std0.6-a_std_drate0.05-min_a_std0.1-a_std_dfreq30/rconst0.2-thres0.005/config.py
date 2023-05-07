# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        'path' : './data/',
        'test_day' : '2022-05-04',
        'split_method' : 'train_validation_test', 
        'timesteps_dim': 3,
        'n_predictions' : 1
    },
    'agent': {
        'train': {
            'episodes': 100,

            'critic_coeff': 0.5,
            'entropy_coeff': 0.01,
            'eps_clip': 0.2,       
            'gamma': 0.01,                
            'batch_size': 800,  
            'lr_actor': 0.0003,       
            'lr_critic': 0.001,      
            'K_epochs': 20, 
            'action_std': 0.6,                   
            'action_std_decay_rate': 0.05,       
            'min_action_std': 0.1,                
            'action_std_decay_freq': int(30) 
        },
        "env" : {
            'transaction_thres' : 0.005,
            'reward_const' : 0.2
        },
        "save" : {
            'save_freq' : 20,
            'base' : './',
            "checkpoints" : "checkpoints/",
            "logs" : "logs/",
            'figs' : "figs/",
            'configs': 'configs/'
        }
    },
}   
