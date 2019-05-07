import cv2
import gym
import gym.spaces
import numpy as np
import collections
from flappy_env import *

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        assert frame.shape == (512, 288, 3)
        img = frame
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = cv2.resize(img, (84, 105), interpolation=cv2.INTER_AREA)
        img = img[:84, :]
        img = np.reshape(img, (84, 84, 1))
        return img.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        h, w, c = env.observation_space.low.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w))

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps=4, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        self.c, self.h, self.w = env.observation_space.low.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n_steps,self.c, self.h, self.w), dtype=dtype)

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

def make_env():
    env = FlappyBird()
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env)
    env = ScaledFloatFrame(env)
    return env