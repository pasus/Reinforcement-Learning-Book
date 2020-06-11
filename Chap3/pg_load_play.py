# PG main

import gym
from pg_agent import PGagent

def main():

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = PGagent(env)

    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
#         env.render()
        action = agent.predict(state)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()