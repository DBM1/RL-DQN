import random
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler


class RolloutStorage(object):
    def __init__(self, config):
        self.obs = torch.zeros([config.max_buff, *config.state_shape], dtype=torch.uint8)
        self.next_obs = torch.zeros([config.max_buff, *config.state_shape], dtype=torch.uint8)
        self.rewards = torch.zeros([config.max_buff, 1])
        self.actions = torch.zeros([config.max_buff, 1])
        self.actions = self.actions.long()
        self.masks = torch.ones([config.max_buff, 1])
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.num_steps = config.max_buff
        self.step = 0
        self.current_size = 0

        self.scale = 5

    def add(self, obs, actions, rewards, next_obs, masks):
        self.obs[self.step].copy_(torch.tensor(obs[None, :], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_obs[self.step].copy_(torch.tensor(next_obs[None, :], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.actions[self.step].copy_(torch.tensor(actions, dtype=torch.float))
        self.rewards[self.step].copy_(torch.tensor(rewards, dtype=torch.float))
        self.masks[self.step].copy_(torch.tensor(masks, dtype=torch.float))
        self.step = (self.step + 1) % self.num_steps
        self.current_size = min(self.current_size + 1, self.num_steps)

    def sample(self, mini_batch_size=None, model=None, target_model=None):
        if model is None or target_model is None:
            indices = np.random.randint(0, self.current_size, mini_batch_size)
        else:
            indices = np.random.randint(0, self.current_size, mini_batch_size * self.scale)
            s0 = self.obs[indices]
            s1 = self.next_obs[indices]
            a = self.actions[indices]
            r = self.rewards[indices]
            done = self.masks[indices]
            if self.config.use_cuda:
                s0 = s0.float().to(self.config.device) / 255.0
                s1 = s1.float().to(self.config.device) / 255.0
                a = a.to(self.config.device)
                r = r.to(self.config.device)
                done = done.to(self.config.device)

            # How to calculate Q(s,a) for all actions
            # q_values is a vector with size (batch_size, action_shape, 1)
            # each dimension i represents Q(s0,a_i)
            all_q_values_model = self.model(s0).cuda()
            all_q_values_target = self.target_model(s1).cuda()

            # How to calculate argmax_a Q(s,a)
            q_values_target = all_q_values_target.max(1)[0].unsqueeze(-1)
            target = (self.config.gamma * (1 - done) * q_values_target) + r
            # Tips: function torch.gather may be helpful
            # You need to design how to calculate the loss
            q_values_model = all_q_values_model.gather(1, a)
            abs_td_error = torch.abs(target - q_values_model).squeeze().numpy()
            top_indices = np.argpartition(abs_td_error, -mini_batch_size)[-mini_batch_size:]
            indices = top_indices
        obs_batch = self.obs[indices]
        obs_next_batch = self.next_obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]
        return obs_batch, obs_next_batch, actions_batch, rewards_batch, masks_batch
