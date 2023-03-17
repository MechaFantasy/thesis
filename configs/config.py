# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        'path' : './data/',
        'val_day' : '2021-07-05', 
        'test_day' : '2022-05-04',
        'seq_len': 60
    },
    'agent': {
        "model": {
            'CNN' : {
                'num_filter': 16
            },
            'LSTM' : {
                'hidden_size' : 16
            }
        },
        'train': {
            'model_type' : 'CNN',
            'gamma' : 0.9,
            'exploration_rate_min' : 0.0001,
            'exploration_rate' : .5,
            'exploration_decay' : 0.9999,
            'replay_memory_size' : 6000,
            'batch_size' : 256,#256
            'lr' : 0.001, #learning_rate
            'episodes': 50,
            'start_from_episode' : 1,
            'sync_every' : 1500, #cap nhat target_net moi gia tri sync_every
            'learn_every' : 30, #hoc policy_net theo gia tri learn_every
            'burnin' : 1000,  #chi hoc policy_net neu relay_memory > burnin
            'patience' : 5
        },
        "save" : {
            'base' : './',
            "checkpoints" : "checkpoints/",
            "logs" : "logs/"
        },
        'env' : {
            'mode' : 'static' #static : reward la hang so, dynamic: reward phu thuoc vao thay doi gia
        }
    },
}   