import numpy as np
import math
import gym

env_name = 'Pendulum-v0'
env = gym.make(env_name)

gains = np.loadtxt('./save_weights/kalman_gain.txt', delimiter=" ")

T = gains[-1, 0]
T = np.int(T)

Kt = gains[:, 1:4]
kt = gains[:, -1]


i_ang = 180.0*np.pi/180.0
x0 = np.array([math.cos(i_ang), math.sin(i_ang), 0])

bad_init = True
while bad_init:
    state = env.reset()  # shape of observation from gym (3,)
    x0err = state - x0
    if np.sqrt(x0err.T.dot(x0err)) < 0.1:  # x0=(state_dim,)
        bad_init = False


for time in range(T+1):
    env.render()

    Ktt = np.reshape(Kt[time, :], [1, 3])
    action = Ktt.dot(state) + kt[time]
    action = np.clip(action, -env.action_space.high[0], env.action_space.high[0])
    ang = math.atan2(state[1], state[0])


    print('Time: ', time, ', angle: ', ang * 180.0 / np.pi, 'action: ', action)

    state, reward, _, _ = env.step(action)

env.close()