#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)


    def step(self, key=None):
        if key is not None:
            self.optimizers[key].step()
        else:
            _ = [self.optimizers[key].step() for key in self.keys]

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]


def build_optimizer(parameters):
    optimizer, scheduler = _define_optimizer(parameters)
    return optimizer, scheduler

def _define_optimizer(params):
    optimizer_params = params['optimizer_params']
    sch_params = params['scheduler_params']
    optimizer = AdamW(
        params['params'],
        lr=optimizer_params.get('lr', 1e-4),
        weight_decay=optimizer_params.get('weight_decay', 5e-4),
        betas=(0.9, 0.98),
        eps=1e-9)
    scheduler = _define_scheduler(optimizer, sch_params)
    return optimizer, scheduler

def _define_scheduler(optimizer, params):
    print(params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 5e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        final_div_factor=5)

    return scheduler

def build_multi_optimizer(parameters_dict, scheduler_params):
    optim = dict([(key, AdamW(params, lr=1e-4, weight_decay=1e-6, betas=(0.9, 0.98), eps=1e-9))
                   for key, params in parameters_dict.items()])

    schedulers = dict([(key, _define_scheduler(opt, scheduler_params)) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim