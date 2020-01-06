# LQR with Fitted Linear Model main

import gym
from lqrflm_agent import LQRFLMagent
import math
import numpy as np
from config import configuration


def main():

    MAX_ITER = 60
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = LQRFLMagent(env)

    agent.update(MAX_ITER)

    T = configuration['T']
    Kt = agent.prev_control_data.Kt
    kt = agent.prev_control_data.kt

    print("\n\n Now play ................")
    # play
    x0 = agent.init_state

    play_iter = 5
    save_gain = []

    for pn in range(play_iter):

        print("     play number :", pn+1)

        if pn < 2:
            bad_init = True
            while bad_init:
                state = env.reset()  # shape of observation from gym (3,)
                x0err = state - x0
                if np.sqrt(x0err.T.dot(x0err)) < 0.1:  # x0=(state_dim,)
                    bad_init = False
        else:
            state = env.reset()

        #state = env.reset()  # shape of observation from gym (3,)

        for time in range(T+1):
            env.render()
            action = Kt[time, :, :].dot(state) + kt[time, :]
            action = np.clip(action, -agent.action_bound, agent.action_bound)
            ang = math.atan2(state[1], state[0])

            print('Time: ', time, ', angle: ', ang*180.0/np.pi, 'action: ', action)

            save_gain.append([time, Kt[time, 0, 0], Kt[time, 0, 1], Kt[time, 0, 2], kt[time, 0]])

            state, reward, _, _ = env.step(action)

    np.savetxt('./save_weights/kalman_gain.txt', save_gain)


if __name__=="__main__":
    main()