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


class deepModel(tf.keras.Model):
    def __init__(self, input_shape, n_actions):
        super(deepModel, self).__init__()
        self.conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape)
        self.conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2,2))
        self.conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2,2))
        self.conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling2D((2,2))
        self.conv5 = Conv2D(filters=256, kernel_size=(3,3), activation='relu')
        self.bn4 = BatchNormalization()
        self.pool4 = MaxPooling2D((2,2))
        self.conv6 = Conv2D(filters=512, kernel_size=(3,3), activation='relu')
        self.bn5 = BatchNormalization()
        self.pool5 = MaxPooling2D((2,2))
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(n_actions, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.pool5(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class shallowModel(tf.keras.Model):
    def __init__(self, input_dims, n_actons):
        super(shallowModel, self).__init__()
        self.conv1 = Conv2D(filters=128, kernel_size=(5,5))
        self.conv2 = Conv2D(filters=256, kernel_size=(5,5))
        self.conv3 = Conv2D(filters=512, kernel_size=(5,5))
        self.act = tf.keras.layers.Activation('relu')
        self.flatten = Flatten()
        self.dense = layers.Dense(n_actons, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.dense(x)
    
