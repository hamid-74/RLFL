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

dataset = 'fmnist'
distribution = 'NIID'

result_folder_name = 'results/RL_test_final/' + dataset + '_' + distribution + '/'


runs = ['run5']


comms_round = 100
# comms_round = 2



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

    
    
    size = len(data)//num_clients


    # print("shard size: ", size)
    
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
    # print(report)
    labels = list(report)
    labels = labels[:-3]
    for label in labels:
        accs.append(report[label]['recall'])
    
    return accs
    # print(len(report.keys()))

def test_categorical (X_test, Y_test,  model):
    label_accuracies = []
    y_pred = model.predict(X_test)

    Y_test = np.argmax(Y_test, axis=1)
    

    for label in range(10):  # Assuming there are 10 labels (0-9)
        # Find indices where the true labels match the current label
        label_indices = np.where(Y_test == label)[0]

        # Extract the true labels and predicted labels for the current label
        true_labels = Y_test[label_indices]

        
        predicted_labels = np.argmax(y_pred[label_indices], axis=1)

        # Calculate accuracy for the current label
        label_accuracy = np.mean(true_labels == predicted_labels)
        label_accuracies.append(label_accuracy)



def ternarize_weights(local_model):

    constant = 1

    quantiles = [0.33, 0.66]
    quantiles_list = []

    for layer in local_model.layers:
        # Get the weights of the layer
        layer_weights = layer.get_weights()

        if len(layer_weights) > 0:

            flattened_weights = [w.flatten() for w in layer_weights]

            # Calculate the quantiles of the weights
            quantiles_values = [np.percentile(flattened_weights, q * 100) for q in quantiles]

            # Append the weights and their quantiles to the respective lists

            quantiles_list.append(quantiles_values)

    # Print the quantiles of weights for each layer
    # for i, quantiles_values in enumerate(quantiles_list):
    #     print(f"Layer {i + 1} - Quantiles of Weights: {quantiles_values}")

    i = 0
    for layer in local_model.layers:
        # Get the weights of the layer
        layer_weights = layer.get_weights()

        if len(layer_weights) > 0:


            # Update the weights based on the specified conditions
            layer_weights = [np.where(w > quantiles_list[i][1], 1.0 * constant, np.where((w >= quantiles_list[i][0]) & (w <= quantiles_list[i][1]), 0.0, -1.0 * constant)) for w in layer_weights]
            # layer_weights = [np.where(w > 0.01, 1.0, np.where((w >= -0.01) & (w <= 0.01), 0.0, -1.0)) for w in layer_weights]
            
            # Set the updated weights back to the layer, leaving biases unchanged
            layer.set_weights(layer_weights)

            i = i + 1

    return local_model.get_weights()


def plot_weight_histogram(model, plot_name):
    all_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            all_weights.extend(weights[0].flatten())  # Weights[0] contains the weight values
    plt.clf()
    plt.hist(all_weights, bins=50, edgecolor='k')
    plt.title('Histogram of Model Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.savefig(plot_name, dpi=300) 


def dump_stats(global_acc_loss_action, global_accs, accs_all_clients, max_acc_achieved,folder_name, run):



    with open(folder_name + 'global_acc_loss_action_' + run +'.json', 'w') as file:
        json.dump(global_acc_loss_action, file)

    
    with open(folder_name + 'global_accs_'  + run +'.json', 'w') as file:
        json.dump(global_accs, file)

    with open(folder_name + 'accs_clients_'  + run +'.json', 'w') as file:
        json.dump(accs_all_clients, file)
    
    with open(folder_name + 'max_acc_'  + run +'.json', 'w') as file:
        json.dump(max_acc_achieved, file)
    



class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, use_bias=False, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200, use_bias=False))
        model.add(Activation("relu"))
        model.add(Dense(classes, use_bias=False))
        model.add(Activation("softmax"))
        return model
    

##loading the dataset
num_classes = 10 

if(dataset=='mnist'):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
elif(dataset=='fmnist'):
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
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

y_train_1234 = keras.utils.np_utils.to_categorical(y_train_1234, num_classes)
y_train_rest = keras.utils.np_utils.to_categorical(y_train_rest, num_classes)

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





###### loading the RL agent
RL_model_name = 'saved_RL_agents/fmnist/keras_fmnist_NIID_20' #hardcoded double cheese burger
# RL_model_name = 'saved_RL_agents/fmnist/keras_fmnist_NIID_980'
RL_agent = tf.keras.models.load_model(RL_model_name)

RL_agent.compile()





lr = 0.01



loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )       


build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST

max_loss = 4


actions = [0, 0.25, 0.5, 1]



for run in runs:

    max_acc_achieved = 0



    smlp_global = SimpleMLP()

    
    global_model = smlp_global.build(build_shape, 10) 

    global_acc_loss_action = []


    accs_all_clients = []
    for i in range(10):
        accs_all_clients.append([])
    global_accs = []

    initial_state = [0] * 100 + [0] * 10 + [0] + [1] + [0]
    # initial_state = [0] + [1] + [0]
    state = list()

    #commence global training loop
    for comm_round in range(comms_round):

        if(comm_round == 0):
            initial_state = np.array(initial_state)
            input_data = np.array(initial_state).reshape(-1, *initial_state.shape)
        else:
            state = np.array(state)
            input_data = np.array(state).reshape(-1, *state.shape)

        # print(input_data)
        value_function = RL_agent.predict(input_data)[0]
        action = np.argmax(value_function)
        print(value_function)
        ternary_scale = actions[action]

        state = list()
        
        
        # if(comm_round < start_point):
        #     ternary_scale = 1
        # else:
        #     ternary_scale = 1

        sum_of_scales = 0

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # plot_weight_histogram(global_model, 'plots/weights' + str(comm_round) + '.png')

        
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

                    
        
        client_id = 0
        #loop through each client and create new local model
        
        # for client in client_names:
        for iii in range(10):
        
            

            
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(build_shape, 10)
            local_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)
            
            accs = []
            
            #set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            test_categorical(X_test, y_test, local_model)

            
            
            #fit local model with client's data

            if(distribution == 'NIID'):
                if(iii<4): 
                    client = client_names_1234[iii]
                    history = local_model.fit(clients_batched_1234[client], epochs=1, verbose=0)
                else: 
                    client = client_names_rest[iii-4]
                    history = local_model.fit(clients_batched_rest[client], epochs=1, verbose=0)

            elif(distribution == 'IID'):
                client = client_names[iii]
                history = local_model.fit(clients_batched[client], epochs=1, verbose=0)


            scaling_factor = 0.1 # weight_scalling_factor(clients_batched, client)

            if(iii < 4):
                local_weights = ternarize_weights(local_model)
                local_model.set_weights(local_weights)
                scaling_factor = ternary_scale * scaling_factor



            sum_of_scales = sum_of_scales + scaling_factor

            for(X_test, Y_test) in test_batched:
                accs = test_model_categorical(X_test, Y_test, local_model)

            accs_all_clients[client_id].append(accs)
            state = state + accs
            
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
        state = state + temp_global_accs




        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:

            print('ternary_scale: {} | max_acc_achieved: {} | run: {} | dataset: {}'.format(ternary_scale, max_acc_achieved,run, dataset))

            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

            if(global_acc > max_acc_achieved):
                max_acc_achieved = global_acc
            
            global_acc_loss_action.append((global_acc, global_loss.numpy().tolist(), ternary_scale))
            # global_loss_list.append()
        state = state + [global_acc] + [global_loss/max_loss] + [(comm_round + 1)/10]
        
        # print(temp_global_accs)


    dump_stats(global_acc_loss_action, global_accs, accs_all_clients, max_acc_achieved, result_folder_name, run)

    K.clear_session()

    
            
            





        
        

