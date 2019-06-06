from dqn_utils import *
import dqn
from wrappers import make_env


def atari_learn(env,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0,                   1e-4 * lr_multiplier),
        (num_iterations / 10,
         1e-4 * lr_multiplier),
        (num_iterations / 2,
         5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return env.get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            # (0, 1.0),
            # (1e6, 0.1),
            (0, 0.4),
            (5e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )
    lr_schedule = dict(milestones=[num_iterations/2], gamma=0.5)

    dqn.learn(
        env=env,
        lr_schedule=lr_schedule,
        load_path='model/step_2400000.pth.tar',
        # load_path=None,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=700000,
        # replay_buffer_size=70000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        # learning_starts=50,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=True
    )
    env.close()


def main():
    # Get Atari games.
    env = make_env()
    # Run training
    atari_learn(env, num_timesteps=2e8)


if __name__ == "__main__":
    main()
