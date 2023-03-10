# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        'path' : './thesis_data/',
        'test_day' : '2022-05-04',
        'seq_len': 20
    },
    'agent': {
        "model": {
            'num_filter': 16,
            'action_dim' : 2
        },
        'train': {
            'gamma' : 0.99,
            'exploration_rate_min' : 0.1,
            'exploration_rate' : 1,
            'exploration_decay' : 0.99,
            'relay_memory' : 15000,
            'batch_size' : 1024,
            'lr' : 1e-6, #learning_rate
            'episodes': 50,
            'sync_every' : 3e3, #cap nhat target_net moi gia tri sync_every
            'save_every' : 3e4, #save model theo gia tri save_every
            'learn_every' : 10, #hoc policy_net theo gia tri learn_every
            'burnin' : 3e3  #chi hoc policy_net neu relay_memory > burnin
        },
        'test' : {
            'load' : 100 #load_model_path
        },
        "save" : {
            "checkpoints" : "checkpoints/",
            "logs" : "logs/"
        },
        'env' : {
            'mode' : 'dynamic' #static : reward la hang so, dynamic: reward phu thuoc vao thay doi gia
        }
    },
}   