import os
import retro
from torchvision import transforms
import gym
from gym.spaces import Box
import gym_super_mario_bros
import numpy as np
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = torch.device("cuda")
from Discretizer import Discretizer
from nearpy import Engine
from nearpy.hashes.custom import CustomHash
from nearpy.filters import NearestFilter, UniqueFilter
import imagehash
from PIL import Image
import math



#Donde se lleva cuenta de los estados por los que hemos pasado y calcula la recompensa que le toca
class RewardEngine():
    def __init__(self):
        self.engine = {}
        self.reward_history = []

    def process_states(self, states):
        total_reward = 0.0
        for s in states:
            tmp_obs = Image.fromarray(s).resize((84, 84)).convert('L')
            encountered = imagehash.phash(tmp_obs, hash_size=4)
            if encountered in self.engine:
                self.engine[encountered] += 1
            else:
                self.engine.setdefault(encountered, 1)

            total_reward += 1/math.sqrt(self.engine[encountered])

        self.reward_history.append(total_reward)
        if len(self.reward_history) > 100:
            del self.reward_history[0]

        return total_reward

    def average_reward(self):
        return np.mean(np.array(self.reward_history))

engine = RewardEngine()


#modificado de https://github.com/dredwardhyde/reinforcement-learning/blob/main/mario_enviroment/mario_ppo.py

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        global engine

        done = False
        states = []
        for i in range(self._skip):
            obs, _, done, info = self.env.step(action)
            states.append(obs)

            if done:
                break

        total_reward = engine.process_states(states)

        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)
    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


env = Discretizer(retro.make('MarioBrosLand2-GameBoy', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))
#env = Discretizer(retro.make('WarioLand', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))
#env = Discretizer(retro.make('MarioBros-Nes2', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))
#env = Discretizer(retro.make('PokemonRed-GameBoy', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))
#env = Discretizer(retro.make('Kirby', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))
#env = Discretizer(retro.make('SuperMarioBros-Nes', inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE,use_restricted_actions=retro.Actions.ALL))


env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)
env.seed(42)
env.action_space.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )
        print(env.action_space.n, 'possible actions')
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).reshape(-1)


class PPOSolver:
    def __init__(self):
        self.rewards = []
        self.gamma = 0.95
        self.lamda = 0.95
        self.worker_steps = 4096
        self.n_mini_batch = 4
        self.epochs = 30
        self.save_directory = "./mario_ppo"
        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.obs = env.reset().__array__()
        self.policy = Model().to(device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.00025},
            {'params': self.policy.critic.parameters(), 'lr': 0.001}
        ], eps=1e-4)
        self.policy_old = Model().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0

    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(self.episode))
        torch.save(self.policy_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))

    def sample(self):
        global engine
        rewards = np.zeros(self.worker_steps, dtype=np.float32)
        actions = np.zeros(self.worker_steps, dtype=np.int32)
        done = np.zeros(self.worker_steps, dtype=bool)
        obs = np.zeros((self.worker_steps, 4, 84, 84), dtype=np.float32)
        log_pis = np.zeros(self.worker_steps, dtype=np.float32)
        values = np.zeros(self.worker_steps, dtype=np.float32)
        engine = RewardEngine()
        for t in range(self.worker_steps):
            with torch.no_grad():
                obs[t] = self.obs
                pi, v = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
            self.obs, rewards[t], done[t], _ = env.step(actions[t])
            self.obs = self.obs.__array__()
            env.render()
            self.rewards.append(rewards[t])

            if engine.average_reward() < 0.1:
                rewards[t] = -100
                done[t] = True


            print(rewards[t])

            if done[t]:
                break


        self.episode += 1
        print('episode', self.episode, 'with reward', np.sum(self.rewards))
        self.all_episode_rewards.append(np.sum(self.rewards))
        self.rewards = []
        env.reset()
        engine = {}

        if self.episode % 10 == 0:
            print('Episode: {}, average reward: {}'.format(self.episode, np.mean(self.all_episode_rewards[-10:])))
            self.all_mean_rewards.append(np.mean(self.all_episode_rewards[-10:]))
            plt.plot(self.all_mean_rewards)
            plt.savefig("{}/mean_reward_{}.png".format(self.save_directory, self.episode))
            plt.clf()
            self.save_checkpoint()
        returns, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }

    def calculate_advantages(self, done, rewards, values):
        _, last_value = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = last_value.cpu().data.numpy()
        values = np.append(values, last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def train(self, samples, clip_range):
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                loss = self.calculate_loss(clip_range=clip_range, samples=mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

    def calculate_loss(self, samples, clip_range):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.policy(samples['obs'])
        ratio = torch.exp(pi.log_prob(samples['actions']) - samples['log_pis'])
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        entropy_bonus = pi.entropy()
        vf_loss = self.mse_loss(value, sampled_returns)
        loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus
        return loss.mean()


solver = PPOSolver()
#solver.load_checkpoint('checkpoint_130.pth')
while True:
    solver.train(solver.sample(), 0.2)
