from agents.VanillaDQN import *


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer

  We can update all Q_nets for every update. However, this makes training really slow.
  Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    self.nets_to_use = cfg['eta']['init_d']
    self.update_d_iterval = cfg['eta']['update_d_interval']
    self.update_d_gamma = cfg['eta']['update_d_gamma']
    self.Q_G_delta = 0

    # Create k different: Q value network, Target Q value network and Optimizer
    self.Q_net = [None] * self.k
    self.Q_net_target = [None] * self.k
    self.optimizer = [None] * self.k
    for i in range(self.k):
      self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()

  def learn(self):
    # Choose a Q_net to udpate
    self.update_Q_net_index = np.random.choice(list(range(self.k)))
    super().learn()
    # Update target network
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
  
  def compute_q_target(self, batch):
    with torch.no_grad():
      q_min = torch.min(torch.stack([self.Q_net_target[i](batch.next_state) for i in range(self.nets_to_use)], 1), 1)
      q_next = q_min.max(1)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_min = torch.min(torch.stack([self.Q_net_target[i](state) for i in range(self.nets_to_use)], 1), 1)
    q_min = to_numpy(q_min).flatten()
    return q_min

  def update_d(self):
    if self.Q_G_delta < 0:
      self.nets_to_use = max(self.nets_to_use - 1, 1)
    if self.Q_G_delta > 0:
      self.nets_to_use = min(self.nets_to_use + 1, self.k)

  def eval_thresholds(self, replay_buffer, n_per_episode):
    res_uniform = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'uniform')
    res_episodes = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'episodes')
    res = dict()
    res.update(res_uniform)
    res.update(res_episodes)
    last_Q_G_delta = res[f'LastReplay_{self.sampling_scheme}/Q_value_t={self.critic.n_quantiles}'] - \
              res[f'LastReplay_{self.sampling_scheme}/Returns']
    self.Q_G_delta = self.Q_G_delta * self.delta_gamma + last_Q_G_delta * (1 - self.delta_gamma)
    return res