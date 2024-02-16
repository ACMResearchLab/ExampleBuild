from agent import Agent 
from environment import StockTradingEnv
import datetime as dt
from datetime import timedelta
import numpy as np
import gym

if __name__ == '__main__':
    env = StockTradingEnv()
    n_days = 20000
    score = 0
    observation = env.reset()
    
    done = True

    agent = Agent(gamma=0.99, epsilon=0.5, alpha=0.005, input_dims=240000, n_actions=3, mem_size=1000, batch_size=64, epsilon_end=0.01)
    # TODO You made dumb dimentions and fucked up the layers in th(e model

    scores = []
    eps_history = []

    for i in range(n_days): 
        action = agent.choose_action(observation)
        observation_,reward, done, info= env.step(action)
        print(f"Day {i}--{info}")
        scores.append(reward)
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        # error = agent.learn()
        # print(er)
        agent.learn()

        eps_history.append(agent.epsilon)

        if i % 1000 == 0 and i > 0 :
            agent.save_model()

    
