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
    self.q_g_eval_interval = cfg['eta']['Q_G_eval_interval']
    self.q_g_n_per_episode = cfg['eta']['Q_G_n_per_episode']
    self.sampling_scheme = cfg['eta']['sampling_scheme']
    self.Q_G_delta = 0
    # tmp action size
    self.replay = BiasControlReplayBuffer(action_dim=1, state_dim=self.state_size, device=self.device, max_size=int(cfg['memory_size']),
                                          gamma=cfg['discount'], n_episodes_to_store=cfg['eta']['Q_G_n_episodes'])
    # Remove TimeLimit from env
    self.max_episode_length = cfg['env']['max_episode_steps']
    self.env = {
      'Train': make_env(cfg['env']['name'], max_episode_steps=int(self.max_episode_length), no_timelimit=True),
      'Test': make_env(cfg['env']['name'], max_episode_steps=int(self.max_episode_length), no_timelimit=True)
    }
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

  def run_episode(self, mode, render):
    while True:
      self.action[mode] = self.get_action(mode)
      if render:
        self.env[mode].render()
      # Take a step
      self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(self.next_state[mode])
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      ep_end = self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length
      if mode == 'Train':
        # Save experience
        self.save_experience(ep_end)
        # Update policy
        if self.time_to_learn():
          self.learn()
        if self.step_count % self.update_d_iterval == 0:
          self.update_d()
        if self.step_count > 10000 and self.step_count % self.q_g_eval_interval == 0:
          res = self.eval_thresholds(self.replay, self.q_g_n_per_episode)
          self.logger.add_scalars('Q_G_stats', res)
        self.step_count += 1
      # Update state
      self.state[mode] = self.next_state[mode]
      if self.done[mode] or self.episode_step_count[mode] >= self.max_episode_length:
        break
    # End of one episode
    self.save_episode_result(mode)
    # Reset environment
    self.reset_game(mode)
    if mode == 'Train':
      self.episode_count += 1

  def save_experience(self, ep_end):
    mode = 'Train'
    self.replay.add(self.state[mode], self.action[mode], self.next_state[mode],
                    self.reward[mode], self.done[mode], ep_end)

  def learn(self):
    # Choose a Q_net to udpate
    self.update_Q_net_index = np.random.choice(list(range(self.k)))
    super().learn()
    # Update target network
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
    if self.show_tb:
      self.logger.add_scalar(f'NumNets', self.nets_to_use, self.step_count)
  
  def compute_q_target(self, batch):
    with torch.no_grad():
      q_min, _ = torch.min(torch.stack([self.Q_net_target[i](batch.next_state) for i in range(self.nets_to_use)], 1), 1)
      q_next = q_min.max(1)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_min, _ = torch.min(torch.stack([self.Q_net[i](state) for i in range(self.nets_to_use)], 1), 1)
    q_min = to_numpy(q_min).flatten()
    return q_min

  def update_d(self):
    if self.Q_G_delta < 0:
      self.nets_to_use = max(self.nets_to_use - 1, 1)
    if self.Q_G_delta > 0:
      self.nets_to_use = min(self.nets_to_use + 1, self.k)

  def eval_thresholds_by_type(self, replay_buffer, n_per_episode, sampling_scheme):
    alpha = torch.exp(self.log_alpha)
    if sampling_scheme == 'uniform':
      states, actions, returns = replay_buffer.gather_returns_uniform(self.discount, float(alpha), n_per_episode)
    elif sampling_scheme == 'episodes':
      states, actions, returns = replay_buffer.gather_returns(self.discount, float(alpha), n_per_episode)
    else:
      raise Exception("No such sampling scheme")
    q = self.get_action_selection_q_values(states).gather(1, actions)
    res = {f'LastReplay_{sampling_scheme}/Q_value': q.mean().__float__(),
           f'LastReplay_{sampling_scheme}/Returns': returns.mean().__float__()}
    return res
  
  def eval_thresholds(self, replay_buffer, n_per_episode):
    res_uniform = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'uniform')
    res_episodes = self.eval_thresholds_by_type(replay_buffer, n_per_episode, 'episodes')
    res = dict()
    res.update(res_uniform)
    res.update(res_episodes)
    last_Q_G_delta = res[f'LastReplay_{self.sampling_scheme}/Q_value'] - \
              res[f'LastReplay_{self.sampling_scheme}/Returns']
    res[f'LastQ_G_delta_{self.sampling_scheme}'] = last_Q_G_delta
    self.Q_G_delta = self.Q_G_delta * self.delta_gamma + last_Q_G_delta * (1 - self.delta_gamma)
    return res
