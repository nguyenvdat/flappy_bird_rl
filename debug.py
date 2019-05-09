# from flappy_env import *
from wrappers import *
import matplotlib.pyplot as plt
import pygame
from actor_critic import *

def test_step_method():
    env = FlappyBird()
    env.reset()
    imgs = []
    cont = True
    while cont:
        FPSCLOCK = pygame.time.Clock()
        events = pygame.event.get()
        for i, event in enumerate(events):
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                cont = False
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                state, reward, done, _ = env.step(1)
                img = Image.fromarray(state)
                imgs.append(img)
                print('key down')
                print(reward)
                break
            elif i == len(events) - 1:
            # else:
                state, reward, done, _ = env.step(0)
                print('key up')
                print(reward)
        pygame.event.clear()
        pygame.display.update()
        FPSCLOCK.tick(30)
    for img in imgs[:1]:
        img.show()

def test_processframe84_wrapper():
    env = FlappyBird()
    env = ProcessFrame84(env)
    state = env.reset()
    assert state.shape == (84, 84, 1), 'processframe84 incorrect shape'
    # plt.imshow(np.squeeze(state), cmap='gray')
    # plt.show()

def test_imagetopytorch_wrapper():
    env = FlappyBird()
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    state = env.reset()
    assert state.shape == (1, 84, 84), 'imagetopytorch incorrect shape'

def test_buffer_wrapper():
    env = FlappyBird()
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    state = env.reset()
    env.step(1)
    # env.step(1)
    state, _, _, _ = env.step(1)
    assert state.shape == (4, 84, 84), 'bufferwrapper incorrect shape'
    plt.imshow(state[1], cmap='gray')
    plt.show()

def test_scaledfloatframe_wrapper():
    env = FlappyBird()
    env = ScaledFloatFrame(env)
    state = env.reset()
    assert state.shape == (512, 288, 3)
    assert np.min(state) >= 0 and np.max(state) <= 1

def test_makeenv():
    env = make_env()
    state = env.reset()
    assert state.shape == (4, 84, 84)
    assert np.min(state) >= 0 and np.max(state) <= 1

def test_network():
    net = Network((1, 84, 84), 2)
    x = torch.zeros(2, 1, 84, 84)
    out = net(x)
    assert out.size() == (2, 2)


if __name__ == "__main__":
    # test_step_method()
    # test_processframe84_wrapper()
    # test_imagetopytorch_wrapper()
    # test_buffer_wrapper()
    # test_scaledfloatframe_wrapper()
    # test_makeenv()
    test_network()