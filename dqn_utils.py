import numpy as np

class ReplayBuffer(object):
    def __init__(self, size, ob_dim):
        self.size = size
        self.idx      = -1
        self.num_in_buffer = 0

        self.obs      = np.empty([self.size] + list(ob_dim), dtype=np.float32)
        self.action   = np.empty([self.size], dtype=np.int32)
        self.reward   = np.empty([self.size], dtype=np.float32)
        self.done     = np.empty([self.size], dtype=np.bool)

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        ids = np.random.choice(self.num_in_buffer - 1, batch_size,                    replace=False)
        return self._encode_sample(ids)

    def recent_observation(self):
        assert self.num_in_buffer >= 1
        return self.obs[self.idx]

    def _encode_sample(self, ids):
        obs_batch      = self.obs[ids]
        act_batch      = self.action[ids]
        rew_batch      = self.reward[ids]
        next_obs_batch = self.obs[ids + 1]
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in                     ids], dtype=np.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def store_effect(self, ob, action, reward, done):
        self.idx = self.idx + 1 if self.idx + 1 < self.size else 0
        self.num_in_buffer = self.num_in_buffer + 1 if self.num_in_buffer + 1 <                      self.size else self.size
        self.obs[self.idx] = ob
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx] = done