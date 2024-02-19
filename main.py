from agent import Agent 
from environment import StockTradingEnv
import datetime as dt
from datetime import timedelta
import numpy as np
import gym

if __name__ == '__main__':
    env = StockTradingEnv()
    n_days = 2000
    score = 0
    observation = env.reset()
    
    done = True
    
    use_original_model = False;
    
    if use_original_model:
        input_dimensions = [200,300,4]
    else:
        input_dimensions = 240000
    
    #control which model via use_v1_model parameter, set to false by default
    agent = Agent(gamma=0.99, epsilon=0.5, alpha=0.005, input_dims=input_dimensions, n_actions=3, mem_size=1000, batch_size=64, epsilon_end=0.01, use_v1_model=use_original_model)
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

    
