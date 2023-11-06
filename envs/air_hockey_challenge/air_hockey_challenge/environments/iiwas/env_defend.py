import numpy as np

from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
        Class for the air hockey defending task.
        The agent should stop the puck at the line x=-0.6.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):
        self.init_velocity_range = np.array([2, 3])

        self.start_range = np.array([[0.7, 0.9], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 0.8], [-0.4, 0.4]])  # Robot Frame
        self.init_angle_range = np.array([-0.2, 0.2])


        self.first_init_velocity_range = np.array([2, 3])

        self.first_start_range = np.array([[0.7, 0.9], [-0.4, 0.4]])  # Table Frame
        self.first_init_ee_range = np.array([[0.60, 0.8], [-0.4, 0.4]])  # Robot Frame
        self.first_init_angle_range = np.array([-0.2, 0.2])



        self.second_init_velocity_range = np.array([2, 5])

        self.second_start_range = np.array([[0.29, 0.9], [-0.4, 0.4]])  # Table Frame
        self.second_init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self.second_init_angle_range = np.array([-0.5, 0.5])





        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, obs):
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(self.init_angle_range[0], self.init_angle_range[1])

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # Puck in Goal
        if (np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if puck_pos[0] > self.env_info['table']['length'] / 2:
                return True

            if puck_pos[0] < -self.env_info['table']['length'] / 2:
                return True

        # Puck stuck in the middle
        if np.abs(puck_pos[0]) < 0.15 and np.linalg.norm(puck_vel[0]) < 0.025:
            return True

        if puck_pos[0] > 0.974 - self.env_info['puck']['radius'] - 0.03 and puck_vel[0] < 0:
            return True
        return super().is_absorbing(state)


if __name__ == '__main__':
    env = AirHockeyDefend()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    env.render()
    while True:
        # action = np.random.uniform(-1, 1, env.info.action_space.low.shape) * 8
        action = np.zeros(7)
        observation, reward, done, info = env.step(action)
        env.render()
        print(observation)
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
