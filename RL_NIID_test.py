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


import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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



def load(paths, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels        
        im_gray = cv2.imread(imgpath , cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten() # cv2.imread(imgpath) 
        # print(image.shape)
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))

    # return a tuple of the data and labels
    
    return data, labels

def create_clients(image_list, label_list, num_clients=100, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)  # <- IID
    
    # sort data for non-iid
#     max_y = np.argmax(label_list, axis=-1)
#     sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
#     data = [(x,y) for _,y,x in sorted_zip]

    #shard data and place at each client
    
    
    size = len(data)//num_clients
    # size = 378

    print("shard size: ", size)
    
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    
    
    if debug:
        print('global_count', global_count, 'local_count', local_count, 'bs', bs)
    
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

def test_model_categorical (X_test, Y_test,  model):
    accs = list()
    y_pred = model.predict(X_test)
    
    report = classification_report (tf.argmax(Y_test, axis=1), tf.argmax(y_pred, axis=1), output_dict=True)
    report2 = classification_report (tf.argmax(Y_test, axis=1), tf.argmax(y_pred, axis=1))
    labels = list(report)
    labels = labels[:-3]
    for label in labels:
        accs.append(report[label]['recall'])
    
    return accs
    # print(len(report.keys()))
    
def subtract_half(original_list):
    modified_list = [x - 0.5 for x in original_list]
    return modified_list


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    




img_path = 'Kaggle-MNIST-Inception-CNN/trainingSet' #'../input/cifar10-pngs-in-folders/cifar10/test'  # <-- test dataset #'../input/mnistasjpg/trainingSample/trainingSample' # <-- smaller dataset
img_path_1234 = 'Kaggle-MNIST-Inception-CNN_1234/trainingSet'
img_path_rest = 'Kaggle-MNIST-Inception-CNN_rest/trainingSet'


#get the path list using the path object
image_paths = list(paths.list_images(img_path))
image_paths_1234 = list(paths.list_images(img_path_1234))
image_paths_rest = list(paths.list_images(img_path_rest))

#apply our function
image_list, label_list = load(image_paths, verbose=10000)
image_list_1234, label_list_1234 = load(image_paths_1234, verbose=10000)
image_list_rest, label_list_rest = load(image_paths_rest, verbose=10000)

#binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

lb = LabelBinarizer()
label_list_1234 = lb.fit_transform(label_list_1234)

lb = LabelBinarizer()
label_list_rest = lb.fit_transform(label_list_rest)


X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                    label_list, 
                                                    test_size=0.1, 
                                                    random_state=42)

X_train_1234, X_test_1234, y_train_1234, y_test_1234 = train_test_split(image_list_1234, 
                                                    label_list_1234, 
                                                    test_size=0.1, 
                                                    random_state=50)
                                            

X_train_rest, X_test_rest, y_train_rest, y_test_rest = train_test_split(image_list_rest, 
                                                    label_list_rest, 
                                                    test_size=0.1, 
                                                    random_state=52)

                                                   
len(X_train), len(X_test), len(y_train), len(y_test)

clients = create_clients(X_train, y_train, num_clients=100, initial='client')

clients_1234 = create_clients(X_train_1234, y_train_1234, num_clients=41, initial='client_1234')

clients_rest = create_clients(X_train_rest, y_train_rest, num_clients=59, initial='client_rest')


clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)


clients_batched_1234 = dict()
for (client_name, data) in clients_1234.items():
    # print(client_name)
    clients_batched_1234[client_name] = batch_data(data)


clients_batched_rest = dict()
for (client_name, data) in clients_rest.items():
    clients_batched_rest[client_name] = batch_data(data)



#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))



input_shape_RL = 10 * 10 + 10 # local accs, previous global acc
num_actions_RL = 5 # 0, 0.25, 0.5, 0.75, 1


RL_model_name = 'keras_v9(khub)'
value_network = tf.keras.models.load_model(RL_model_name)






epsilon_RL = 0
    
alpha_RL = 0.1
episode_RL = 0




lr = 0.01
# comms_round = 2
# comms_round = 300
comms_round = 150


loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )       


build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST



runs = [  'run16', 'run17','run18',]
# runs = [ 'run38', 'run39', 'run40', 'run41', 'run42', 'run43']

actions = [0, 0.25, 0.5, 0.75, 1]

for run in runs:

    

    #10 no of clients
    




    smlp_global = SimpleMLP()
    global_model = smlp_global.build(build_shape, 10) 
    global_acc_list = []
    global_loss_list = []
    all_actions = []

    accs_all_clients = []
    for i in range(10):
        accs_all_clients.append([])
    global_accs = []

    state = []

    #commence global training loop
    for comm_round in range(comms_round):


        sum_of_scales = 0

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        
        #initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        #randomize client data - using keys
        all_client_names = list(clients_batched.keys())
        all_client_names_1234 = list(clients_batched_1234.keys())
        all_client_names_rest = list(clients_batched_rest.keys())


        # print(all_client_names_rest)   
        client_names = random.sample(all_client_names, k=10)
        client_names_1234 = random.sample(all_client_names_1234, k=4)
        client_names_rest = random.sample(all_client_names_rest, k=6)
        # print(client_names, len(client_names))
        random.shuffle(client_names)
        random.shuffle(client_names_1234)
        random.shuffle(client_names_rest)
        
    #     if debug: 
    #         # print('all_client_names', all_client_names)
    #         print('client_names', client_names, len(client_names))
                    
        counter = 0
        client_id = 0

        comm_round_actions = []
        #loop through each client and create new local model
        
        
        # for client in client_names:
        for iii in range(10):
        
            

            
            
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(build_shape, 10)
            local_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)
            
            
            
            #set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            
            #fit local model with client's data

            if(iii<4): 
                client = client_names_1234[iii]
                local_model.fit(clients_batched_1234[client], epochs=1, verbose=0)
            else: 
                client = client_names_rest[iii-4]
                local_model.fit(clients_batched_rest[client], epochs=1, verbose=0)

            print(client)
            
            # local_model.fit(clients_batched[client], epochs=1, verbose=0)

            scaling_factor = 0.1 # weight_scalling_factor(clients_batched, client)

            local_weights = local_model.get_weights()
            

            
            num_ones = 0
            num_mones = 0
            num_zeros = 0
            
            

            if(counter < 4):
                # print("Ternary:")
                # print("before: ")
                # for(X_test, Y_test) in test_batched:       
                #     local_acc, local_loss = test_model(X_test, Y_test, local_model, comm_round)

                
                

                for i in range(0,local_weights[0].shape[0]):
                    for j in range(0,local_weights[0].shape[1]):
                        
                    
                        if(local_weights[0][i,j] > 0.01):
                            local_weights[0][i,j] = 1
                            num_ones = num_ones + 1
                        elif(local_weights[0][i,j] < -0.01):
                            local_weights[0][i,j] = -1 
                            num_mones = num_mones + 1
                        else:
                            local_weights[0][i,j] = 0
                            num_zeros = num_zeros + 1

                for i in range(0,local_weights[2].shape[0]):
                    for j in range(0,local_weights[2].shape[1]):
                        
                    
                        if(local_weights[2][i,j] > 0.01):
                            local_weights[2][i,j] = 1
                            num_ones = num_ones + 1
                        elif(local_weights[2][i,j] < -0.01):
                            local_weights[2][i,j] = -1 
                            num_mones = num_mones + 1
                        else:
                            local_weights[2][i,j] = 0
                            num_zeros = num_zeros + 1

                for i in range(0,local_weights[4].shape[0]):
                    for j in range(0,local_weights[4].shape[1]):
                        
                    
                        if(local_weights[4][i,j] > 0.01):
                            local_weights[4][i,j] = 1
                            num_ones = num_ones + 1
                        elif(local_weights[4][i,j] < -0.01):
                            local_weights[4][i,j] = -1
                            num_mones = num_mones + 1
                        else:
                            local_weights[4][i,j] = 0
                            num_zeros = num_zeros + 1
                
                counter = counter + 1
                

                # print(num_ones, num_mones, num_zeros)
                # print (local_weights)

            
                local_model.set_weights(local_weights)

                
            
                # print(local_model.get_weights())
                
                # print("after: ")
                # for(X_test, Y_test) in test_batched:
                #     local_acc, local_loss = test_model(X_test, Y_test, local_model, comm_round)

            

            for(X_test, Y_test) in test_batched:
                accs = test_model_categorical(X_test, Y_test, local_model)

            accs_all_clients[client_id].append(accs)
            
            #scale the model weights and add to list
            
            # print('scaling_factor', scaling_factor)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            client_id = client_id + 1
            
            #clear session to free memory after each communication round
            K.clear_session()
        
        ############# making the next state and doing batch training
        next_state = []

        if(comm_round>1): 

            for iiii in range(10):
                next_state = next_state + subtract_half(accs_all_clients[iiii][comm_round])
                
            next_state = next_state + subtract_half(global_accs[comm_round - 1])
            next_state = np.array(next_state)

       
        #############
        
        state = next_state
        

        if(comm_round > 2):
            input_data = np.array(state).reshape(-1, *state.shape)
            value_function = value_network.predict(input_data)[0]
            action = np.argmax(value_function)
            action_to_take = actions[action]
            # main_action_to_take = action_to_take * ((0.995)**(comm_round))
            # if(main_action_to_take<0.5):
            #     main_action_to_take=0.5
            # action_to_take = 0.25

            # if(action == 4):
            #     action = 3
            # elif(action==1 or action ==0):
            #     action = 2

            
           

        else:
            action = 2
            action_to_take = 0.5
            # main_action_to_take = 0.5
            
            
            # action_to_take = 0.01
        
        
        
        
        sum_of_scales = 0.6 + (action_to_take * 0.4)
        # sum_of_scales = 0.6 + (main_action_to_take * 0.4)

        
         
        #to get the average over all the local model, we simply take the sum of the scaled weights
        for iiii in range(4):
            # print('before:', scaled_local_weight_list[iiii])
            scaled_local_weight_list[iiii] = scale_model_weights(scaled_local_weight_list[iiii], action_to_take)
            # scaled_local_weight_list[iiii] = scale_model_weights(scaled_local_weight_list[iiii], main_action_to_take)
            # print('after:', scaled_local_weight_list[iiii])

        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        average_weights = scale_model_weights(average_weights, 1/sum_of_scales)

        #update global model 
        global_model.set_weights(average_weights)

        temp_global_accs = []
        for(X_test, Y_test) in test_batched:
            temp_global_accs = test_model_categorical(X_test, Y_test, local_model)

        global_accs.append(temp_global_accs)

        all_actions.append(action_to_take)

        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            print(run, RL_model_name)
            print('action:', action_to_take)
            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
            global_acc_list.append(global_acc)
            global_loss_list.append(global_loss)

        #fix reward and done
        reward = global_acc_list[comm_round] - global_acc_list[comm_round - 1]
        if(global_acc > 0.90):
            done = 1
        else:
            done = 0
        ##########
        





    # IID 
    
    # plt.figure(figsize=(16,4))
    # plt.subplot(121)
    # plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
    # plt.subplot(122)
    # plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
    # print('IID | total comm rounds', len(global_acc_list))

    # plt.show()
    
    with open('results/RL/accs_all_clients_NIID_RL_' + run + '_' + RL_model_name + '.json', 'w') as file:
        json.dump(accs_all_clients, file)

    with open('results/RL/global_accs_NIID_RL_' + run + '_' + RL_model_name + '.json', 'w') as file:
        json.dump(global_accs, file)

    with open('results/RL/actions_RL_' + run + '_' + RL_model_name + '.json', 'w') as file:
        json.dump(all_actions, file)

    iid_df = pd.DataFrame(list(zip(global_acc_list, global_loss_list)), columns=['global_acc_list', 'global_loss_list'])
    iid_df.to_csv('results/RL/MNIST_NIID_RL_'  + run + '_' + RL_model_name +'.csv',index=False)


