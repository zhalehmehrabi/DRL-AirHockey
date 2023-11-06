import os

from gym import Env
from gym.spaces import Box, Discrete, Tuple
import numpy as np
import random
# from envs.humanoid_clipped import HumanoidClipped


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
            min_max_norm=True 
    ):
        ProxyEnv.__init__(self, env)
        self.min_max_norm = min_max_norm 
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def to_n1p1(self, state): 
        v_min = self._wrapped_env.observation_space.low
        v_max = self._wrapped_env.observation_space.high
        if any(v_min == - np.inf) or any(v_max ==np.inf):
            raise ValueError('unbounded state')
        new_min, new_max = -1, 1
        res = (state - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        res = np.nan_to_num(res, nan=0) # if we want to keep z at zero
        res = np.clip(res, new_min, new_max)
        return res

    def get_opt_action(self, state):
        a = self._wrapped_env.get_opt_action(self.from_n1p1(state))
        a = self.to_n1p1(a) # this function shuold be used for states only
        return a

    def from_n1p1(self, state): 
        v_min = self._wrapped_env.observation_space.low
        v_max = self._wrapped_env.observation_space.high
        new_min, new_max = -1, 1
        res = (state - new_min)/(new_max - new_min)*(v_max - v_min) + v_min
        res =  np.nan_to_num(res, nan=0)
        return res

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def reset(self, **kwargs):
        state = super().reset() # np array
        if self.min_max_norm:
            return self.to_n1p1(state)
        else:
            return state

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        if self.min_max_norm and not self._should_normalize: # added
            next_obs = self.to_n1p1(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


def domain_to_env(name):
    # try:
    # from gym.envs.mujoco import HalfCheetahEnv, \
    #     InvertedPendulumEnv, HumanoidEnv, \
    #     HopperEnv, AntEnv, Walker2dEnv
    # from envs.cartpole_continuous import ContinuousCartPoleEnv as Cartpole
    # from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv as MountainCar
    #from gym.envs.box2d.bipedal_walker import BipedalWalker
    # from envs.river_swim_continuous import RiverSwimContinuous
    # from envs.point import PointEnv
    # from envs.humanoid_clipped import HumanoidClipped
    # from envs.hopper_clipped import HopperClipped
    # from envs.walker_clipped import WolkerClipped
    # from envs.cheetah_clipped import HalfCheetahEnvClipped
    # from envs.ant_maze import AntMazeEnv
    # from envs.lqg1d import LQG1D
    # from envs.cliff_continuous_mono import CliffWorldContinuousMono
    # from envs.cliff_continuous import CliffWorldContinuous
    from envs.air_hockey import AirHockeyEnv
    return {
        # 'invertedpendulum': InvertedPendulumEnv,
        # #'humanoid': HumanoidEnv,
        # 'humanoid': HumanoidClipped,
        # 'halfcheetah': HalfCheetahEnvClipped,
        # 'hopper': HopperClipped,
        # 'ant': AntEnv,
        # 'ant_maze': AntMazeEnv,
        # 'walker2d': WolkerClipped,
        # 'cartpole': Cartpole,
        # 'mountain': MountainCar,
        # 'riverswim': RiverSwimContinuous,
        # 'point': PointEnv,
        # 'lqg': LQG1D,
        # 'cliff': CliffWorldContinuous,
        # 'cliff_mono': CliffWorldContinuousMono,
        'air_hockey':AirHockeyEnv
    }[name]
    # except:
    #     from envs.cartpole_continuous import ContinuousCartPoleEnv as Cartpole
    #     from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv as MountainCar
    #     from envs.river_swim_continuous import RiverSwimContinuous
    #     from envs.point import PointEnv
    #     from envs.lqg1d import LQG1D
    #     from envs.cliff_continuous_mono import CliffWorldContinuousMono
    #     from envs.cliff_continuous import CliffWorldContinuous
    #     from envs.air_hockey import AirHockeyEnv
    #     #from gym.envs.box2d.bipedal_walker import BipedalWalker
    #     return {
    #         'cartpole': Cartpole,
    #         'mountain': MountainCar,
    #         'riverswim': RiverSwimContinuous,
    #         'lqg': LQG1D,
    #         'cliff': CliffWorldContinuous,
    #         'cliff_mono': CliffWorldContinuousMono,
    #         'point': PointEnv,
    #         'air_hockey': AirHockeyEnv
    #     }[name]


def domain_to_epoch(name):
    return {
        'invertedpendulum': 300,
        'humanoid': 9000,
        'halfcheetah': 5000,
        'hopper': 2000,
        'ant': 5000,
        'walker2d': 5000,
        'bipedal': 2000,
        'cartpole': 300,
        'mountain': 200,
        'riverswim': 200,
        'lqg': 200,
        'cliff': 200,
        'cliff_mono': 200,
        'point': 400,
        'air_hockey':300
    }[name]


def env_producer(domain, seed, **args):
    env = domain_to_env(domain)(**args)
    # seed = random.randint(0, 999999)

    env.seed(seed)
    env = NormalizedBoxEnv(env)

    return env

def create_producer(domain="airhockey", seed=0,**env_args):

    if seed==0:
        seed = random.randint(0, 999999)
    # seed = random.randint(0, 999999)

    return lambda: env_producer(domain, seed, **env_args)
