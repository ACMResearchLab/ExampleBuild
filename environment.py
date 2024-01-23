import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import matplotlib.image as mpimg
import os
from candles import getBarData
import glob

class StockTradingEnv(Env):
    def __init__(self):
        # Initialize variables

        #Actions we can take: Buy, Sell, Hold
        self.action_space = Discrete(3)
        #Observationspace wlil be created
        self.observation_space = Box(low=0, high=255, shape=(300, 200, 3))

        self.state = 0
        self.day= 1
        
        self.current_stock = ''

        self.profit = 0
        self.purchase_price = 0
        self.action_taken_before = 0

        self.multiplier = 1


    def step(self, action):
        # firsly we get the observation



        # 0 is hold 1 is buy and 2 is sell
        # Now we will do an if statement 
        # We will return info as well
        # if action == 0:

        path_str = f"Candle Data/Training Data/{self.day}*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])

        

        array_splt = img_path[0].split(",")
        stock_name = array_splt[2]
        stock_name = stock_name.split(".")
        current_stock = stock_name[0]
        current_price = float(array_splt[1])  


        multiplier_reset = False
        done = False
        if(self.day == 1):
            # return  profit and everythng as zero except of the observation
            reward = 0
            info  = f"The EQUITY is changing to: {self.current_stock}"
            self.current_stock = self.current_stock
        else:
            if(action == 0 ):
                if(self.action_taken_before == 0):
                    reward = 0
                    info  = "No action was taken"
                else:
                    reward = 0
                    info  = f"{current_stock} is HELD with action: {self.action_taken_before}"
            elif(action == 1):
                if(self.action_taken_before == 0):
                    reward = 0
                    info  = f"{current_stock} was BOUGHT at price: {round(current_price)}"
                    self.purchase_price = current_price
                    self.action_taken_before = 1
                elif(self.action_taken_before == 1):
                    reward = 0
                    info  = f"{current_stock} is BOUGHT AGAIN"
                    self.multiplier = self.multiplier + 1
                elif(self.action_taken_before == 2): #Meaning the stock was sold before and now the trade is exitted
                    reward = self.purchase_price - current_price
                    info = f"The trade EXITS (short) at {round(current_price)} profit: {round(reward * self.multiplier,2)}"
                    multiplier_reset = True
                    done = True
                    self.action_taken_before = 0
            elif(action == 2):
                if(self.action_taken_before == 0):
                    reward = 0
                    info  = f"{current_stock} was SOLD at price: {round(current_price)}"
                    self.purchase_price = current_price
                    self.action_taken_before = 2
                elif(self.action_taken_before == 1): #Meaning the stock was sold before and now the trade is exitted
                    reward = current_price - self.purchase_price
                    info = f"The trade EXITS (Long) at {round(current_price)} profit: {round(reward * self.multiplier,2)}"
                    multiplier_reset = True
                    done = True
                    self.action_taken_before = 0
                elif(self.action_taken_before == 2): 
                    reward = 0
                    info  = f"{current_stock} is SOLD AGAIN"
                    self.multiplier = self.multiplier + 1



        
        self.profit = self.profit + reward*self.multiplier
        info = f"{info} ----- Total Revenue: {round(self.profit, 2)}"
    
        self.day = self.day +1
        # self.purchase_price = current_price
        reward = reward*self.multiplier

        if(multiplier_reset == True):
            self.multiplier = 1
        return img,reward,done, info

    def reset(self):
        # img = mpimg.imread('images/TrainingDS/1,3216.3,ASIANPAINT.png')
        path_str = f"Candle Data/Training Data/{self.day},*"
        img_path = glob.glob(path_str)
        img = mpimg.imread(img_path[0])
        # now we have the image and the path

        array_splt = img_path[0].split(",")
        self.current_price = float((array_splt[1]))

        self.current_stock = (array_splt[1].split("."))[0]
        
        # self.previous_price = self.current_price
        self.day = 1

        # return self.current_price
        self.profit = 0
        return img

    
