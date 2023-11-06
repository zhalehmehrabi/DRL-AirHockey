from copy import deepcopy

from air_hockey_challenge.constraints import *
from air_hockey_challenge.environments import position_control_wrapper as position
from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.utils.kinematics import forward_kinematics
from mushroom_rl.core import Environment
import torch
import random
import numpy as np

class AirHockeyChallengeWrapper(Environment):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,

            "7dof-hit": position.IiwaPositionHit,
            "7dof-defend": position.IiwaPositionDefend,
            "7dof-prepare": position.IiwaPositionPrepare,

            "3dof-hit": position.PlanarPositionHit,
            "3dof-defend": position.PlanarPositionDefend
        }

        if env == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        self.base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info
        self.half_success = 0

        if custom_reward_function:
            self.base_env.reward = lambda state, action, next_state, absorbing: custom_reward_function(self.base_env,
                                                                                                       state, action,
                                                                                                       next_state,
                                                                                                       absorbing)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        if "7dof" in self.env_name or self.env_name == "tournament":
            constraint_list.add(LinkConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults
            info["constr_reward"] = 1

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)
            info["unsuccess"] = self.check_unsuccess(obs)
            if self.half_success == 0:
                self.half_success = self.check_half_success(obs)

            info["half_success"] = self.half_success
            info["constr_reward"] = 1

        return obs, reward, done, info

    def render(self):
        self.base_env.render()

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.base_env.env_info['joint_pos_ids']]

    def get_ee_pose(self, obs):
        """
        Get the End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            end-effector's position

        """
        res = forward_kinematics(self.base_env.robot_model, self.base_env.robot_data, self.get_joint_pos(obs))
        return res[0]
    def reset(self, state=None):
        # seed = 10
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        s = self.base_env.reset(state)
        self.half_success = 0
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        return s

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        ee_pos = self.get_ee_pose(obs)

        ee_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=ee_pos)
        success = 0
        distance = np.linalg.norm(puck_pos[:2] - ee_pos[:2])

        if "hit" in self.env_name:
            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            # if (distance <= (self.base_env.env_info["puck"]["radius"] + self.base_env.env_info["mallet"]["radius"]) * 1.2
            #         and np.linalg.norm(puck_vel[:2]) > 2) and puck_vel[0] > 2 and puck_vel[1] > 1:

            # if (distance <= (self.base_env.env_info["puck"]["radius"] + self.base_env.env_info["mallet"]["radius"]) * 1.2
            #     and puck_vel[0]) > 2:
            #     success = 1

            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "prepare" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and np.abs(puck_pos[1]) < 0.39105 and puck_vel[0] < 0.1:
                success = 1
        return success

    def check_unsuccess(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        ee_pos = self.get_ee_pose(obs)

        ee_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=ee_pos)
        unsuccess = 0

        if "defend" in self.env_name:
            if ((np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0 and
                    puck_pos[0] < -self.env_info['table']['length'] / 2):
                unsuccess = 1

        return unsuccess

    def check_half_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        # puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        # ee_pos = self.get_ee_pose(obs)
        #
        # ee_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=ee_pos)
        success = 0
        # distance = np.linalg.norm(puck_pos[:2] - ee_pos[:2])
        goal_center_point = np.array([2.484, 0.0])
        if "defend" in self.env_name:
            if np.linalg.norm(puck_pos[:2] - goal_center_point) < 0.3 and puck_vel[0] > 0:
                success = 1
        return success

if __name__ == "__main__":
    env = AirHockeyChallengeWrapper(env="7dof-hit")
    env.reset()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])) * 3
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
