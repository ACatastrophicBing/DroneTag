class Hyperparameters:
    #  Using the same hyperparameters from Josh's setup,
    #  probably will change a couple of these values to modify learning
    def __init__(self):
        self.gamma = 0.995
        self.tau = 0.005
        self.alpha = 0.001
        self.beta = 0.002
        self.lr = 1e-4
        self.ee_lr = 0.01
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-5

        self.intn_epsilon = 1.0
        self.intn_eps_min = 0.01
        self.intn_eps_dec = 1e-5

        self.batch_size = 100
        self.mem_size = 100000
        self.replace_target_ctr = 1000
        self.failed = False
        self.failure_action = [0, 0, 1]

        self.noise = 0.1
        self.update_actor_iter = 2
        self.warmup = 1000
        self.time_step = 0

        self.min_max_action = 1

        #GNN
        self.hidden_channels = 32
        self.num_heads = 4