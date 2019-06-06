import cv2
import gym
import gym.spaces
import numpy as np
import collections
from flappy_env import *


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        assert frame.shape == (512, 288, 3)
        img = frame
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        img = cv2.resize(img, (84, 105), interpolation=cv2.INTER_AREA)
        img = img[:84, :]
        img = np.reshape(img, (84, 84, 1))
        return img.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        h, w, c = env.observation_space.low.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        self.c, self.h, self.w = env.observation_space.low.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_steps*self.c, self.h, self.w), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros((self.n_steps, self.c, self.h, self.w))
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return np.reshape(self.buffer, (self.n_steps*self.c, self.h, self.w))


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32)/255.0
        # return np.array(obs).astype(np.float32)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=2):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
        self.env = env

    def step(self, action):
        total_reward = 0
        for i in range(self._skip):
            if i == 0 and action == constant.UP_ACTION:
                ob, reward, done, _ = self.env.step(constant.UP_ACTION)
            else:
                ob, reward, done, _ = self.env.step(constant.NORMAL_ACTION)
            self._obs_buffer.append(ob)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        # return ob, total_reward, done, _
        return max_frame, total_reward, done, _

    def reset(self):
        self._obs_buffer.clear()
        ob = self.env.reset()
        self._obs_buffer.append(ob)
        return ob


class Monitor(gym.Wrapper):
    def __init__(self, env=None):
        super(Monitor, self).__init__(env)
        self.current_episode_reward = 0
        self.episode_rewards = []
        self.total_step = 0

    def step(self, action):
        ob, reward, done, _ = self.env.step(action)
        self.current_episode_reward += reward
        self.total_step += 1
        if done:
            self.episode_rewards.append(self.current_episode_reward)
        return ob, reward, done, _

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_total_steps(self):
        return self.total_step

    def reset(self):
        ob = self.env.reset()
        self.current_episode_reward = 0
        return ob


def make_env():
    env = FlappyBird()
    env = MaxAndSkipEnv(env, 2)
    env = ProcessFrame84(env)
    # env = ImageToPyTorch(env)
    # env = BufferWrapper(env)
    # env = ScaledFloatFrame(env)
    env = Monitor(env)
    return env
