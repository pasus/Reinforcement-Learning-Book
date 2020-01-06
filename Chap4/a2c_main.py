# A2C main

import gym
from a2c_agent import A2Cagent

def main():

    max_episode_num = 1000
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = A2Cagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()