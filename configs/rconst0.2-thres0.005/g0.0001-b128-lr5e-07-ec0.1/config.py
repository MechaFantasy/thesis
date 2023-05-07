# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        'path' : './data/',
        'test_day' : '2022-05-04',
        'split_method' : 'train_validation_test', 
        'timesteps_dim': 30,
        'n_predictions' : 1
    },
    'agent': {
        'train': {
            'gamma' : 0.0001,
            'batch_size' : 128,
            'lr' : 5e-7, 
            'entropy_coeff' : 0.1,
            'episodes': 50,
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
            'figs' : "figs/"
        }
    },
}   
