import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
import time

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

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class Agent:
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']

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
            action_logit = self.critic_nn(ob)
            return action_logit
        else:
            mean = self.critic_nn(ob)
            log_std = self.log_std
        return (mean, log_std)
    
    def sample_action(self, policy_parameter):
        """
        Sample a random action according to the distribution specified by policy_parameter
        arguments:
            for discrete action: logits of categorical distribution over actions
                logits: (bs)
            for continuous action: (mean, log_std) of a Gaussian distribution over actions 
                mean: (bs, action_dim)
                log_std: action_dim
        returns:
            sample_ac:
                if discrete: (bs)
                if continuous: (bs, action_dim)
        """
        if self.discrete:
            sampled_ac = torch.multinomial(policy_parameter)
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
            loss = nn.CrossEntropyLoss()
            log_prob = -loss(policy_parameter, taken_action)
        else:
            mean, log_std = policy_parameter
            cov = torch.eye(self.ac_dim)
            cov[range(self.ac_dim), range(self.ac_dim)] = torch.exp(log_std) ** 2
            m = MultivariateNormal(mean, cov)
            log_prob = m.log_prob(taken_action)
        return log_prob

    def update_actor(self, ob, ac, adv):
        self.actor_optim.zero_grad()
        policy_parameter = self.get_policy_parameter(ob)
        log_prob = self.get_log_prob(policy_parameter, ac)
        loss = torch.sum(-log_prob * adv)
        loss.backward()
        self.actor_optim.step()

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
        for target_update_count in range(self.num_target_updates):
            # recompute target values
            v_s_next = self.critic_nn(next_obs)
            target_value = (1 - terminal) * (re + self.gamma * v_s_next) + terminal * re
            for grad_step_count in range(self.num_grad_steps_per_target_update):
                self.critic_optim.zero_grad()
                v_s_prediction = self.critic_nn(obs)
                loss_fn = nn.MSELoss()
                loss = loss_fn(v_s_prediction, target_value.detach()) 
                loss.backward()
                self.critic_optim.step()

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += self.pathlength(path)
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
    

    def pathlength(self, path):
        return len(path["reward"])