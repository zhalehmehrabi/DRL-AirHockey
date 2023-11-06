import os
from gym import utils
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import time
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500
    
    
class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(-np.ones_like(low), np.ones_like(high), dtype=np.float32)

        self.action_range = 0.5 * (high - low)
        self.action_center = 0.5 * (high + low)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            try:
                self.sim.step()
            except Exception as e:
                next_obs = self._get_obs()
                print("Error!")
                print(next_obs)
                print(ctrl)
                print(e)
                exit()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer().render()

            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data[:, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    @property
    def action_bounds(self):
        return self.action_space.low, self.action_space.high

class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'point.xml', 2)
        utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        qpos = next_obs[:2]
        goal = [25.0, 0.0]
        reward = -np.linalg.norm(goal - qpos)
        return next_obs, reward, False, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent


if __name__ == "__main__":
    env = PointEnv()
    ob = env.reset()
    print(env.action_space)
    env.render()
    input()