
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import MultivariateNormal
import time
import logz
import time
import inspect
import os
from wrappers import make_env
import matplotlib.pyplot as plt
import queue
from actor_critic import Network
from dqn_utils import *
import copy
import uuid
import pickle
import sys


class QLearner:
    def __init__(
            self,
            env,
            lr_schedule,
            load_path=None,
            exploration=LinearSchedule(1000000, 0.1),
            stopping_criterion=None,
            replay_buffer_size=1000000,
            batch_size=32,
            gamma=0.99,
            learning_starts=50000,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            grad_norm_clipping=10,
            rew_file=None,
            double_q=True):
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        h, w, c = env.observation_space.shape
        ob_dim = (c * frame_history_len, h, w)
        ac_dim = env.action_space.n
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.env = env
        self.exploration = exploration
        self.grad_norm_clipping = grad_norm_clipping
        self.rew_file = str(uuid.uuid4()) + \
            '.pkl' if rew_file is None else rew_file
        self.double_q = double_q

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_ob = self.env.reset()
        self.q_nn = Network(ob_dim, ac_dim)
        self.target_q_nn = Network(ob_dim, ac_dim)
        self.q_nn.to(self.device)
        self.target_q_nn.to(self.device)
        self.replay_buffer = ReplayBuffer(
            replay_buffer_size, frame_history_len)
        self.gamma = gamma
        self.q_nn_optim = torch.optim.Adam(self.q_nn.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.q_nn_optim, **lr_schedule)
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.log_every_n_steps = 10000
        self.num_param_updates = 0

        self.start_time = None
        self.t = 0
        self.load_path = load_path
        if self.load_path is not None:
            self.load_model(self.load_path)

    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

    def step_env(self):
        with torch.no_grad():
            idx = self.replay_buffer.store_frame(self.last_ob)
            recent_obs = self.replay_buffer.encode_recent_observation()
            recent_obs = self.move_channel_axis_and_scale(recent_obs)
            recent_obs = torch.tensor(recent_obs).to(self.device)
            q_val = torch.squeeze(self.q_nn(recent_obs[None, :]))
            policy_action = torch.argmax(q_val).data.cpu().numpy()
            if np.random.uniform() < self.exploration.value(self.t):
                actions = np.arange(self.env.action_space.n)
                actions = np.delete(actions, policy_action)
                action = np.random.choice(actions)
            else:
                action = policy_action
            next_ob, reward, done, _ = self.env.step(action)
            self.replay_buffer.store_effect(idx, action, reward, done)
            if done:
                next_ob = self.env.reset()
            self.last_ob = next_ob
        self.t += 1

    def update_model(self):
        if self.t > self.learning_starts and self.t % self.learning_freq == 0 and self.replay_buffer.can_sample(self.batch_size):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(
                self.batch_size)
            obs_batch = self.move_channel_axis_and_scale(obs_batch)
            next_obs_batch = self.move_channel_axis_and_scale(next_obs_batch)
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
            # act_batch = torch.from_numpy(act_batch).float().to(device)
            next_obs_batch = torch.from_numpy(
                next_obs_batch).float().to(self.device)
            rew_batch = torch.from_numpy(rew_batch).float().to(self.device)
            done_mask = torch.from_numpy(done_mask).float().to(self.device)

            next_q_val = self.q_nn(next_obs_batch)
            next_target_q_val = self.target_q_nn(next_obs_batch)
            if self.double_q:
                max_action_next_q_val = torch.argmax(next_q_val, dim=1)
                next_v = self.gamma * \
                    next_target_q_val[range(self.batch_size),
                                      max_action_next_q_val]  # (bs, )
            else:
                next_v = self.gamma * \
                    torch.max(next_target_q_val, dim=1)[0]  # (bs, )
            y_t = rew_batch + next_v * (1 - done_mask)  # (bs, )
            q_val = self.q_nn(obs_batch)[range(
                self.batch_size), act_batch]  # (bs, )

            self.q_nn_optim.zero_grad()
            l = nn.SmoothL1Loss()
            loss = l(q_val, y_t.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.q_nn.parameters(), self.grad_norm_clipping)
            self.q_nn_optim.step()
            self.lr_scheduler.step()

            if self.num_param_updates % self.target_update_freq == 0:
                self.target_q_nn = copy.deepcopy(self.q_nn)
            self.num_param_updates += 1

    def log_progress(self):
        episode_rewards = self.env.get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])

        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(
                self.best_mean_episode_reward, self.mean_episode_reward)

        if self.t % self.log_every_n_steps == 0:
            print("Timestep %d" % (self.t,))
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % self.exploration.value(self.t))
            print("learning_rate %f" % self.get_lr(self.q_nn_optim))
            if self.start_time is not None:
                print("running time %f" %
                      ((time.time() - self.start_time) / 60.))
            print()

            self.start_time = time.time()

            sys.stdout.flush()

            with open(self.rew_file, 'wb') as f:
                pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_model(self):
        if self.t % 100000 == 0:
            ckpt_dict = {
                'q_nn_state': self.q_nn.cpu().state_dict(),
                'target_q_nn_state': self.target_q_nn.cpu().state_dict(),
                'q_nn_optim_state': self.q_nn_optim.state_dict(),
                't': self.t,
                'num_param_updates': self.num_param_updates,
                # 'replay_buffer': self.replay_buffer,
                'lr_scheduler_state': self.lr_scheduler.state_dict(),
                'best_mean_episode_reward': self.best_mean_episode_reward,
                'env': self.env
            }
            checkpoint_path = os.path.join(
                'model/', 'step_{}.pth.tar'.format(self.t))
            print('Saving model at: ' + checkpoint_path)
            torch.save(ckpt_dict, checkpoint_path)
            # replay_buffer_path = os.path.join(
            #     'model/', 'buffer_step_{}.pth.tar'.format(self.t))
            # with open(replay_buffer_path, 'wb') as output:
            #     pickle.dump(self.replay_buffer, output,
            #                 pickle.HIGHEST_PROTOCOL)
            print('save completed')
            self.q_nn.to(self.device)
            self.target_q_nn.to(self.device)

    def load_model(self, checkpoint_path):
        print('Loading model at: ' + checkpoint_path)
        ckpt_dict = torch.load(checkpoint_path)
        self.q_nn.load_state_dict(ckpt_dict['q_nn_state'])
        self.target_q_nn.load_state_dict(ckpt_dict['target_q_nn_state'])
        self.q_nn_optim.load_state_dict(ckpt_dict['q_nn_optim_state'])
        self.t = ckpt_dict['t']
        self.num_param_updates = ckpt_dict['num_param_updates']
        # self.replay_buffer = ckpt_dict['replay_buffer']
        self.env = ckpt_dict['env']
        self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler_state'])
        self.best_mean_episode_reward = ckpt_dict['best_mean_episode_reward']

    def move_channel_axis_and_scale(self, ob):
        new_ob = ob.astype(np.float32)/255.0
        new_ob = np.moveaxis(new_ob, -1, -3)
        return new_ob


def learn(*args, **kwargs):
    alg = QLearner(*args, **kwargs)
    while not alg.stopping_criterion_met():
        alg.step_env()
        alg.update_model()
        alg.log_progress()
        alg.save_model()
