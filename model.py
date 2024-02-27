import tensorflow as tf
from keras.layers import Dense, Activation, Conv2D, Input, Flatten
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, TimeDistributed , Conv2D, MaxPooling2D
from tensorflow.keras import layers
import numpy as np
import sys



class DeepNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(DeepNet, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), padding='same')
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.conv2_a = Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.conv3_a = Conv2D(filters=128, kernel_size=(3,3), padding='same')
        self.conv4 = Conv2D(filters=256, kernel_size=(3,3), padding='same')
        self.conv5 = Conv2D(filters=512, kernel_size=(3,3), padding='same')
        self.act = tf.keras.layers.Activation('relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.bn_5 = tf.keras.layers.BatchNormalization()

        self.pool = MaxPooling2D((2,2))
        self.add = tf.keras.layers.Add()
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.flatten = Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.fin = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        
        #first conv block
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        
        
        #first res block                
        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.act(x)
        
        prev_x = x 
                
        x = self.conv2_a(x)
        x = self.bn_2(x)
        x = self.add([x, prev_x])
        x = self.act(x)
        
        x = self.pool(x)
        
        #second res block         
        x = self.conv3(x)
        x = self.bn_3(x)
        x = self.act(x)
        
        prev_x = x
        
        x = self.conv3_a(x)
        x = self.bn_3(x)
        x = self.add([x, prev_x])
        x = self.act(x)
        
        x = self.pool(x)       
        
        #final conv layers 
        x = self.conv4(x)
        x = self.bn_4(x)
        x = self.act(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = self.bn_5(x)
        x = self.act(x)  
        
        x = self.global_pool(x)
        
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.fin(x)
        
        return x 
