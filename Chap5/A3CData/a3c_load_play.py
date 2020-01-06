# A3C main

import gym
from a3c_agent import A3Cagent

def main():

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = A3Cagent(env_name)

    agent.global_actor.load_weights('./save_weights/')
    agent.global_critic.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.global_actor.predict(state)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()