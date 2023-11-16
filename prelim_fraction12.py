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

dataset = 'mnist'

fraction = 1

result_folder_name = 'results/prelim_results/fraction/' + dataset + '/'


runs = ['run1']


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
    


    #shard data and place at each client
    
    
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

        # print(f'Accuracy for label {label}: {label_accuracy:.4f}')





    # print(len(report.keys()))  

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


def dump_stats(global_acc_loss, accs_all_clients, max_acc_achieved, folder_name, run, fraction):

    with open(folder_name + 'global_acc_loss_' + str(fraction) + '_' + run + '.json', 'w') as file:
        json.dump(global_acc_loss, file)

    
    with open(folder_name + 'global_accs_'  + str(fraction) + '_' + run + '.json', 'w') as file:
        json.dump(global_accs, file)

    with open(folder_name + 'accs_clients_' + str(fraction) + '_' + run + '.json', 'w') as file:
        json.dump(accs_all_clients, file)
    
    with open(folder_name + 'max_acc_'  + str(fraction) + '_' + run +'.json', 'w') as file:
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
    



num_classes = 10 

if(dataset=='mnist'):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
elif(dataset=='fmnist'):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255





# Convert class vectors to binary class matrices. This is called one hot encoding.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)



clients = create_clients(X_train, y_train, num_clients=100, initial='client')



clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)





#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


lr = 0.01

comms_round = 100
# comms_round = 2


loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )       


build_shape = 784 # 784 for MNIST






for run in runs:

    max_acc_achieved = 0



    smlp_global = SimpleMLP()

    
    global_model = smlp_global.build(build_shape, 10) 

    global_acc_loss = []


    accs_all_clients = []
    for i in range(10):
        accs_all_clients.append([])
    global_accs = []

    #commence global training loop
    for comm_round in range(comms_round):

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # plot_weight_histogram(global_model, 'plots/weights' + str(comm_round) + '.png')

        
        #initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        #randomize client data - using keys
        all_client_names = list(clients_batched.keys())



        # print(all_client_names_rest)   
        client_names = random.sample(all_client_names, k=10)

        # print(client_names, len(client_names))
        random.shuffle(client_names)


                    
        
        client_id = 0

        scaling_factor = 0.1

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
            client = client_names[iii]
            local_model.fit(clients_batched[client], epochs=1, verbose=0)

            # print("fraction: {}".format(fraction*10))

            if(iii < (fraction * 10)):
                local_weights = ternarize_weights(local_model)
                local_model.set_weights(local_weights)


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
        averaged_weights = sum_scaled_weights(scaled_local_weight_list)
        

        #update global model 
        global_model.set_weights(averaged_weights)

        temp_global_accs = []
        for(X_test, Y_test) in test_batched:
            temp_global_accs = test_model_categorical(X_test, Y_test, local_model)

        global_accs.append(temp_global_accs)




        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:

            print('Testing Fraction | max_acc_achieved:: {} | run: {} | dataset: {}'.format( max_acc_achieved, run, dataset))

            global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

            if(global_acc > max_acc_achieved):
                max_acc_achieved = global_acc
            
            global_acc_loss.append((global_acc, global_loss.numpy().tolist()))
            # global_loss_list.append()
        
        # print(temp_global_accs)


    dump_stats(global_acc_loss, accs_all_clients, max_acc_achieved, result_folder_name, run, fraction)

    K.clear_session()

    
            
            





        
        

