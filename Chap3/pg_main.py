# PG main

import gym
from pg_agent import PGagent

def main():

    max_episode_num = 1000
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = PGagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()