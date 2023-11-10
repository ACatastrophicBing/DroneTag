from Actor import Actor

import numpy as np
import math

import torch.nn as nn

class DroneAgent(Actor):
    def __init__(self, n_obs, n_actions, options_per_action, id, learning_scheme, n_chars,
                 meta_param_size = 1, seq_len=5, edge_index = None):