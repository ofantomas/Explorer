from agents.SAC import *


class OffRPG(SAC):
  '''
  Implementation of OffRPG (Off-policy Reward Policy Gradient): DDPG style
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set optimizer for reward function
    self.optimizer['reward'] = getattr(torch.optim, cfg['optimizer']['name'])(self.network.reward_params, **cfg['optimizer']['critic_kwargs'])
  
  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      input_size = self.cfg['feature_dim']
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env[mode].game.state_shape()[2], feature_dim=input_size)
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=input_size)
    elif input_type == 'feature':
      input_size = self.state_size
      feature_net = nn.Identity()
    # Set actor network
    assert self.action_type == 'CONTINUOUS', f"{self.cfg['agent']['name']} only supports continous action spaces."
    actor_net = MLPGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[self.action_size], hidden_act=self.cfg['hidden_act'], rsample=True)
    # Set critic network (state value)
    critic_net = MLPCritic(layer_dims=[input_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set reward network
    reward_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorVCriticRewardNet(feature_net, actor_net, critic_net, reward_net)
    return NN

  def learn(self):
    mode = 'Train'
    batch = self.replay.sample(['state', 'action', 'reward', 'mask', 'next_state'], self.cfg['batch_size'])
    # Take an optimization step for critic
    critic_loss = self.compute_critic_loss(batch)
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    # Take an optimization step for reward
    reward_loss = self.compute_reward_loss(batch)
    self.optimizer['reward'].zero_grad()
    reward_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.reward_params, self.gradient_clip)
    self.optimizer['reward'].step()
    # Take an optimization step for actor
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['actor_update_frequency'] == 0:
      # Freeze reward network to avoid computing gradients for it
      for p in self.network.reward_net.parameters():
        p.requires_grad = False
      # Compute actor loss
      actor_loss = self.compute_actor_loss(batch)
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze reward network
      for p in self.network.reward_net.parameters():
        p.requires_grad = True
      # Update target networks by polyak averaging (soft update)
      self.soft_update(self.network, self.network_target)
      # Log
      if self.show_tb:
        self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
        self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
        self.logger.add_scalar(f'reward_loss', reward_loss.item(), self.step_count)

  def compute_critic_loss(self, batch):
    v = self.network.get_state_value(batch.state)
    with torch.no_grad():
      v_next = batch.mask * self.network_target.get_state_value(batch.next_state).detach()
    critic_loss = (batch.reward + self.discount * v_next - v).pow(2).mean()
    return critic_loss

  def compute_reward_loss(self, batch):
    true_reward = batch.reward
    predicted_reward = self.network.get_reward(batch.state, batch.action)
    reward_loss = (predicted_reward - true_reward).pow(2).mean()
    return reward_loss

  def compute_actor_loss(self, batch):
    with torch.no_grad():
      adv = batch.mask * self.network_target.get_state_value(batch.next_state) - self.network_target.get_state_value(batch.state) / self.discount
      adv = adv.detach()
    repara_action = self.network.get_repara_action(batch.state, batch.action)
    predicted_reward = self.network.get_reward(batch.state, repara_action)
    new_log_pi = self.network.get_log_pi(batch.state, batch.action)
    actor_loss = -(predicted_reward + self.discount * adv * new_log_pi).mean()
    return actor_loss