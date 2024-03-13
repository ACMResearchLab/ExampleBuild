from agent import Agent 
from environment import StockTradingEnv
import datetime as dt
from datetime import timedelta
import numpy as np
import gym

if __name__ == '__main__':
    n_days = 2000
    score = 0
    path = "../ExampleBuild/model_results.txt"
    n_iters = 10
    done = True
    
    #Implmentation of changing model via main:
    models = ['deep', 'shallow', 'deepnet']
    #model = models[1] #Change which element of the list for which model you would like to run
    
    #If the model is the shallow (original) model then we need use the shape [200,300,4] otherwise 240000
    for model in models:
        for iter in range(n_iters):
            env = StockTradingEnv()
            observation = env.reset()
            if model == 'shallow':
                input_dimensions = [200,300,4]
                epochs = 1
            else:
                input_dimensions = 240000
                epochs = 5
            
            agent = Agent(gamma=0.9, epsilon=0.96, alpha=0.05, input_dims=input_dimensions, model_type=model, epochs=epochs, n_actions=3, mem_size=1000, batch_size=64, epsilon_end=0.01)
            # TODO You made dumb dimentions and fucked up the layers in th(e model
            scores = []
            eps_history = []
            #intial var for tracking
            starting_equity = 0;
            final_revenue = 0;
            max_gain = float('-inf')
            max_loss = float('inf')
            max_rev = float('-inf')
            min_rev = float('inf')
            
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
                
                if i == 63:
                    starting_equity = env.profit
                final_revenue = env.profit
                max_gain = max(max_gain, reward)
                max_loss = min(max_loss, reward)
                max_rev = max(max_rev, env.profit)
                min_rev = min(min_rev, env.profit)
        
                try:
                    if i % 1000 == 0 and i > 0 :
                        agent.save_model()
                except Exception as e:
                    print(f"Couldn't save the model becauseeee {e}")
                    
                
            
                
            with open(path, "a") as f:
                f.write(f'\nModel: {model}, Iteration {iter}\n')
                f.write(f'Starting Eq: {starting_equity}\n')
                f.write(f'Final Eq: {final_revenue}\n')
                f.write(f'Max Gain Revenue: {max_rev}\n')
                f.write(f'Lowest Point of Revenue: {min_rev}\n')
                f.write(f'Max Single Trade Gain: {max_gain}\n')
                f.write(f'Max Single Trade Loss: {max_loss}\n')
                print("iter done saved info")
            

    
