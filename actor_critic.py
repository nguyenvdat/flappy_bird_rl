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
from torch.distributions.categorical import Categorical

class Network(nn.Module):
    """
    Construct conv net to recognize game image
    arguments:
        input_shape: shape of observation (n_channel, height, width)
        n_actions: number of actions
    returns:
        for discrete action:
            logits for each action
        for gaussian action:
            mean of the action
    """
    def __init__(self, ob_dim, action_dim):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ob_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(ob_dim)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.conv.apply(self.init_weight)
        self.fc.apply(self.init_weight)
    
    def init_weight(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, is_softmax=False):
        x = torch.as_tensor(x)
        conv_out = self.conv(x).view(x.size()[0], -1)
        if is_softmax:
            return F.softmax(self.fc(conv_out), dim=-1)
        return self.fc(conv_out)

class Agent:
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']
        self.save_path = computation_graph_args['save_path']
        self.load_path = computation_graph_args['load_path']
        self.max_checkpoints = computation_graph_args['max_checkpoints']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']
        self.actor_nn = Network(self.ob_dim, self.ac_dim)
        self.critic_nn = Network(self.ob_dim, 1)
        self.log_std = nn.Parameter(torch.FloatTensor(self.ac_dim))
        self.critic_optim = torch.optim.Adam(self.critic_nn.parameters())
        if self.discrete:
            actor_param_list = self.actor_nn.parameters()
        else:
            actor_param_list = list(self.actor_nn.parameters()) + list(self.log_std)
        self.actor_optim = torch.optim.Adam(actor_param_list)
        self.ckpt_paths = queue.Queue()
        if self.load_path:
            self.step, self.best_val = self.load_model(self.load_path)
        else:
            self.step = 0
            self.best_val = -1


    def get_policy_parameter(self, ob):
        """ 
        Compute the parameters for action given this observation, which are parameters of the policy distribution p(a|s)
        arguments:
            ob: (bs, self.ob_dim)
        return:
            if discrete: logits of categorical distribution over action 
                action_logit: (bs, action_dim)
            if continuous: tuple (mean, log_std) of a Gaussian
                mean: (bs, action_dim)
                log_std: (action_dim) (trainable variable, not output of nn)
        """
        if self.discrete:
            action_logit = self.actor_nn(ob, True)
            return action_logit
        else:
            mean = self.actor_nn(ob)
            log_std = self.log_std
        return (mean, log_std)
    
    def sample_action(self, policy_parameter):
        """
        Sample a random action according to the distribution specified by policy_parameter
        arguments:
            for discrete action: logits of categorical distribution over actions
                logits: (bs, action_dim)
            for continuous action: (mean, log_std) of a Gaussian distribution over actions 
                mean: (bs, action_dim)
                log_std: action_dim
        returns:
            sample_ac:
                if discrete: (bs)
                if continuous: (bs, action_dim)
        """
        if self.discrete:
            logits = policy_parameter
            sampled_ac = torch.multinomial(F.softmax(logits, dim=1), 1)
        else:
            mean, log_std = policy_parameter
            z = torch.randn(self.ac_dim)
            sampled_ac = mean + torch.exp(log_std) * z
        return sampled_ac

    def get_log_prob(self, policy_parameter, taken_action):
        """ 
        Compute the log probability of the taken_action under the current parameters of the policy
        arguments:
            policy_parameters
                if discrete: logits of a categorical distribution over actions
                    logits: (bs, action_dim)
                if continuous: (mean, log_std) of a Gaussian distribution over actions
                    mean: (bs, action_dim)
                    log_std: (action_dim)
            taken_action: (bs) if discrete, (bs, action_dim) if continuous
        returns:
            log_prob: (bs)
        """
        if self.discrete:
            logits = policy_parameter
            bs, _ = logits.size()
            log_prob_v = F.log_softmax(logits, dim=1)
            log_prob = log_prob_v[range(bs), taken_action]
        else:
            mean, log_std = policy_parameter
            cov = torch.eye(self.ac_dim)
            cov[range(self.ac_dim), range(self.ac_dim)] = torch.exp(log_std) ** 2
            m = MultivariateNormal(mean, cov)
            log_prob = m.log_prob(taken_action)
        return log_prob

    def update_actor(self, obs, actions, adv):
        """
            Update parameters of the policy.

            arguments:
                obs: (sum_of_path_lengths, ob_dim)
                actions: (sum_of_path_lengths)
                adv: (sum_of_path_lengths)
            returns:
                nothing
        """
        self.actor_optim.zero_grad()
        policy_parameter = self.get_policy_parameter(obs)
        log_prob = self.get_log_prob(policy_parameter, actions)
        loss = torch.mean(-log_prob * adv)
        print('Previous log_prob: ' + str(-loss))
        loss.backward()
        self.actor_optim.step()
        policy_parameter = self.get_policy_parameter(obs)
        log_prob = self.get_log_prob(policy_parameter, actions)
        loss = torch.mean(-log_prob * adv)
        print('Updated log_prob: ' + str(-loss))

    def update_critic(self, obs, next_obs, re, terminal):
        """ 
        Update the parameters of the critic
        arguments:
            obs: (sum_of_path_lengths, ob_dim) 
            next_obs: (sum_of_path_lengths, ob_dim) 
            re: (sum_of_path_lengths)
            terminal: (sum_of_path_lengths)
        returns: nothing
        """
        for _ in range(self.num_target_updates):
            # recompute target values
            v_s_next = torch.squeeze(self.critic_nn(next_obs))
            target_value = re + self.gamma * v_s_next * (1 - terminal)
            for _ in range(self.num_grad_steps_per_target_update):
                self.critic_optim.zero_grad()
                v_s_prediction = torch.squeeze(self.critic_nn(obs))
                loss_fn = nn.MSELoss()
                loss = loss_fn(v_s_prediction, target_value.detach()) 
                loss.backward()
                self.critic_optim.step()

    def estimate_advantage(self, obs, next_obs, re, terminal):
        """
            Estimates the advantage function value for each timestep.
            arguments:
                obs: (sum_of_path_lengths, ob_dim) 
                next_obs: (sum_of_path_lengths, ob_dim) 
                re: (sum_of_path_lengths)
                terminal: (sum_of_path_lengths)
            returns:
                adv: (sum_of_path_lengths)
        """
        v_s = torch.squeeze(self.critic_nn(obs))
        v_s_next = torch.squeeze(self.critic_nn(next_obs))
        q = re + self.gamma * v_s_next * (1 - terminal)
        adv = q - v_s

        if self.normalize_advantages:
            mean = torch.mean(adv)
            std = torch.std(adv)
            adv = (adv - mean)/(std + 1e-8)
        return adv

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        """
        sample trajectory for one episode, finish when done return by env is 1 or n_steps > self.max_path_length
        """
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            policy_parameter = self.get_policy_parameter(ob[None, :])
            ac = self.sample_action(policy_parameter)
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)


            # add the observation after taking a step to next_obs
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            # YOUR CODE HERE
            # print(steps)
            # plt.imshow(ob[1], cmap='gray')
            # plt.savefig ('image/grafico01' + str(steps) + '.png')
            # plt.show()
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation" : np.array(obs, dtype=np.float32),
                "reward" : np.array(rewards, dtype=np.float32),
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    def save_model(self, step, val):
        ckpt_dict = {
            'actor_state': self.actor_nn.cpu().state_dict(),
            'critic_state': self.critic_nn.cpu().state_dict(),
            'step': step,
            'best_val': self.best_val
        }
        checkpoint_path = os.path.join(self.save_path, 'step_{}.pth.tar'.format(step))
        torch.save(ckpt_dict, checkpoint_path)
        self.ckpt_paths.put(checkpoint_path)
        print('Saved checkpoint: {}'.format(checkpoint_path))
        if self.best_val < val:
            print('New best checkpoint at step {}...'.format(step))
            self.best_val = val
        # remove checkpoint with lower value
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                print('Removed checkpoint: {}'.format(worst_ckpt))
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

    def load_model(self, checkpoint_path):
        ckpt_dict = torch.load(checkpoint_path)
        self.actor_nn.load_state_dict(ckpt_dict['actor_state'])
        self.critic_nn.load_state_dict(ckpt_dict['critic_state'])
        step = ckpt_dict['step']
        best_val = ckpt_dict['best_val']
        return  step, best_val

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def train_AC(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate,
        logdir,
        normalize_advantages,
        seed,
        save_path,
        load_path,
        max_checkpoints,
        save_every):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    # env = gym.make(env_name)
    env = make_env()

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    # env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec['max_episode_steps']

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        'save_path': save_path,
        'load_path': load_path,
        'max_checkpoints': max_checkpoints,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args) #estimate_return_args

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(agent.step + 1, agent.step + 1 + n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        obs = np.concatenate([path["observation"] for path in paths])
        actions = np.concatenate([path["action"] for path in paths])
        re = np.concatenate([path["reward"] for path in paths])
        next_obs = np.concatenate([path["next_observation"] for path in paths])
        terminal = np.concatenate([path["terminal"] for path in paths])
        print(actions[:20])
        # obs = torch.from_numpy(obs).type(torch.float32)
        # actions = torch.from_numpy(actions).type(torch.int8)
        # re = torch.from_numpy(re).type(torch.float32)
        # next_obs = torch.from_numpy(next_obs).type(torch.float32)

        # Call tensorflow operations to:
        # (1) update the critic, by calling agent.update_critic
        # (2) use the updated critic to compute the advantage by, calling agent.estimate_advantage
        # (3) use the estimated advantage values to update the actor, by calling agent.update_actor
        # YOUR CODE HERE
        obs = torch.as_tensor(obs)
        actions = torch.as_tensor(actions).type(torch.long)
        next_obs = torch.as_tensor(next_obs)
        re = torch.as_tensor(re)
        terminal = torch.as_tensor(terminal)
        agent.update_critic(obs, next_obs, re, terminal)
        adv = agent.estimate_advantage(obs, next_obs, re, terminal)
        agent.update_actor(obs, actions, adv.detach())

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()

        # save model
        if itr % save_every == 0:
            agent.save_model(itr, np.mean(returns))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='flappy')
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    # parser.add_argument('--batch_size', '-b', type=int, default=200)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    # parser.add_argument('--num_target_updates', '-ntu', type=int, default=1)
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    # parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=1)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1),
    parser.add_argument('--save_path', type=str, default='./save/'),
    parser.add_argument('--load_path', type=str, default=None),
    parser.add_argument('--max_checkpoints', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=5)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'ac_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_AC(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                num_target_updates=args.num_target_updates,
                num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                seed=seed,
                save_path=args.save_path,
                load_path=args.load_path,
                max_checkpoints=args.max_checkpoints,
                save_every=args.save_every
                )
        train_func()
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

if __name__ == "__main__":
    main()