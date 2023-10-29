import numpy as np
import pandas as pd
import random
import cv2
import os
import keras
from tqdm import tqdm

import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


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
    print('shard size:', size)
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
    



class SimpleMLP:
    @staticmethod
    def build(shape):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model

        

    






num_classes = 10 

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
print(y_test[3])
print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



X_train_1234 = list()
y_train_1234 = list()
X_train_rest = list()
y_train_rest = list()

for i in range(len(X_train)):
    if (y_train[i] == 0 or y_train[i] == 1 or y_train[i] == 2 or y_train[i] == 3):
        X_train_1234.append(X_train[i])
        y_train_1234.append(y_train[i])
    else:
        X_train_rest.append(X_train[i])
        y_train_rest.append(y_train[i])


# Convert class vectors to binary class matrices. This is called one hot encoding.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train_1234 = keras.utils.to_categorical(y_train_1234, num_classes)
y_train_rest = keras.utils.to_categorical(y_train_rest, num_classes)



build_shape = 784



clients = create_clients(X_train, y_train, num_clients=132, initial='client')


clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    
#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))




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


# build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST



runs = [   'run1']
# runs = ['run1']

for run in runs:

    

    #10 no of clients
    


    ternary_factors = [0.25, 0.5, 0.75, 1]


    for ternary_scale in ternary_factors:
        smlp_global = SimpleMLP()
        global_model = smlp_global.build(build_shape) 
        global_acc_list = []
        global_loss_list = []

        accs_all_clients = []
        for i in range(10):
            accs_all_clients.append([])
        global_accs = []

        

        #commence global training loop
        for comm_round in range(comms_round):
            

            sum_of_scales = 0
                    
            # get the global model's weights - will serve as the initial weights for all local models
            global_weights = global_model.get_weights()
            
            #initial list to collect local model weights after scalling
            scaled_local_weight_list = list()

            #randomize client data - using keys
            all_client_names = list(clients_batched.keys())
                
            client_names = random.sample(all_client_names, k=10)
            # print(client_names, len(client_names))
            random.shuffle(client_names)
            
        #     if debug: 
        #         # print('all_client_names', all_client_names)
        #         print('client_names', client_names, len(client_names))
                        
            counter = 0
            client_id = 0
            #loop through each client and create new local model
            for client in client_names:
                smlp_local = SimpleMLP()
                local_model = smlp_local.build(build_shape)
                local_model.compile(loss=loss, 
                            optimizer=optimizer, 
                            metrics=metrics)
                
                accs = []
                
                #set local model weight to the weight of the global model
                local_model.set_weights(global_weights)
                
                #fit local model with client's data
                local_model.fit(clients_batched[client], epochs=1, verbose=0)

                scaling_factor = 0.1 # weight_scalling_factor(clients_batched, client)

                local_weights = local_model.get_weights()
                

                
                num_ones = 0
                num_mones = 0
                num_zeros = 0
                
                

                if(counter < -1):
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

                    scaling_factor = ternary_scale * scaling_factor
                

                    # print(local_model.get_weights())
                    
                    # print("after: ")
                    # for(X_test, Y_test) in test_batched:
                    #     local_acc, local_loss = test_model(X_test, Y_test, local_model, comm_round)

                sum_of_scales = sum_of_scales + scaling_factor

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
                
            #to get the average over all the local model, we simply take the sum of the scaled weights
            average_weights = sum_scaled_weights(scaled_local_weight_list)
            
            average_weights = scale_model_weights(average_weights, 1/sum_of_scales)

            #update global model 
            global_model.set_weights(average_weights)

            temp_global_accs = []
            for(X_test, Y_test) in test_batched:
                temp_global_accs = test_model_categorical(X_test, Y_test, local_model)

            global_accs.append(temp_global_accs)

            #test global model and print out metrics after each communications round
            for(X_test, Y_test) in test_batched:
                print(ternary_scale, run)
                global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
                global_acc_list.append(global_acc)
                global_loss_list.append(global_loss)





        # IID 
        
        # plt.figure(figsize=(16,4))
        # plt.subplot(121)
        # plt.plot(list(range(0,len(global_loss_list))), global_loss_list)
        # plt.subplot(122)
        # plt.plot(list(range(0,len(global_acc_list))), global_acc_list)
        # print('IID | total comm rounds', len(global_acc_list))

        # plt.show()
        
        with open('results/fmnist/categorical_RL/accs_all_clients_' + str(ternary_scale) + '_' + run + '.json', 'w') as file:
            json.dump(accs_all_clients, file)

        with open('results/fmnist/categorical_RL/global_accs_' + str(ternary_scale) + '_' + run + '.json', 'w') as file:
            json.dump(global_accs, file)

        iid_df = pd.DataFrame(list(zip(global_acc_list, global_loss_list)), columns=['global_acc_list', 'global_loss_list'])
        iid_df.to_csv('results/fmnist/categorical_RL/MNIST_IID_' + str(ternary_scale) + '_' + run + '.csv',index=False)


