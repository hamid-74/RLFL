import numpy as np
import pandas as pd
import random
import cv2
import os
import keras
import math
from tqdm import tqdm

from collections import deque
import random
import pickle

from tensorflow.keras.layers import Dense, LeakyReLU


import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.utils import np_utils


import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from imutils import paths

debug = 0
dataset = "fmnist"









input_shape_RL = 10 * 10 + 10 + 1 + 1 + 1  # local accs, previous global accs, previous global accuracy, 

# input_shape_RL = 3
#previous global loss/4, comm round [0 ta 100]/100
num_actions_RL = 4 # 0, 0.25, 0.5, 1


#originally 4 * input_shape_RL

print('creating a new model')
value_network = tf.keras.models.Sequential([
tf.keras.layers.Dense(100, activation='relu', input_shape=(input_shape_RL,)),
tf.keras.layers.Dense(100, activation='relu'),
tf.keras.layers.Dense(num_actions_RL)
])


optimizer_RL = tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer_RL = tf.keras.optimizers.Adam(learning_rate=0.002)
loss_fn_RL = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")





if(os.path.exists('results/merged_data/' + dataset + '.pkl')):
    with open('results/merged_data/' + dataset + '.pkl', 'rb') as file:
        replay_RL = pickle.load(file)
else:
    print('error!!!')

print(len(replay_RL))

for i in range(len(replay_RL)):
    replay_RL[i][0][-1] = replay_RL[i][0][-1] * 10
#     if(i==98):    print(replay_RL[i][3][-1])
    # replay_RL[i][3][-1] = replay_RL[i][3][-1] * 10



print(replay_RL[0])

    
# alpha_RL = 0.1
alpha_RL = 1
episode_RL = 0
# gamma_RL = 0.95
gamma_RL = 0.5
batch_RL = 2000











actions = [0, 0.25, 0.5, 1]



for epoch in range(1000):
    
    #replay_RL.append((state,action,reward,next_state,done))


    
    if len(replay_RL)>batch_RL:
        with tf.GradientTape() as tape:
            batch_ = random.sample(replay_RL,batch_RL)
            q_value1 = value_network((tf.convert_to_tensor([x[0] for x in batch_])))

            q_value2 = value_network((tf.convert_to_tensor([x[3] for x in batch_])))
            
            reward = tf.convert_to_tensor([x[2] for x in batch_])
            action = tf.convert_to_tensor([x[1] for x in batch_])
            done =   0

            
            actual_q_value1 = tf.cast(reward,tf.float64) + tf.cast(tf.constant(alpha_RL),tf.float64)*(tf.cast(tf.constant(gamma_RL),tf.float64)*tf.cast((tf.constant(1)-done),tf.float64)*tf.cast(tf.reduce_max(q_value2),tf.float64))           
            loss_RL = tf.cast(tf.gather(q_value1,action,axis=1,batch_dims=1),tf.float64)
            loss_RL = loss_RL - actual_q_value1
            loss_RL = tf.reduce_mean(tf.math.pow(loss_RL,2))

    
            grads = tape.gradient(loss_RL, value_network.trainable_variables)
            optimizer_RL.apply_gradients(zip(grads, value_network.trainable_variables))

            print('Episode {} Epoch {} done with loss {} !!!!!!'.format(episode_RL, epoch,loss_RL))
            episode_RL += 1

            if(epoch%20 == 0):   
                alpha_RL = alpha_RL * 0.99
                value_network.save('saved_RL_agents/'+ dataset +'/keras_' + dataset +'_NIID_' + str(epoch) + '/')
            
            




