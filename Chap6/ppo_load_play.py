# PPO with GAE Load and Play

import gym
from ppo_agent import PPOagent

def main():

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    agent = PPOagent(env)

    agent.actor.load_weights('./save_weights/')
    agent.critic.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.actor.predict(state)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()