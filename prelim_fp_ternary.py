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

result_folder_name = 'results/prelim_results/fp_ternary/'

# runs = [  'run1', 'run2' ,'run3','run4','run5']
runs = [   'run1'] # 4 5 to do
start_point = 10

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




def plot_weight_histogram_three_regions(model, plot_name):
    all_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            all_weights.extend(weights[0].flatten())  # Weights[0] contains the weight values

    num_regions = 3
    elements_per_region = len(all_weights) // num_regions


    # print('elements per region: {}'.format(elements_per_region))

    
    all_weights.sort()
    region_boundaries = [0] * (num_regions - 1)

    for i in range(1, num_regions):
        region_boundaries[i - 1] = all_weights[i * elements_per_region]


    colors = ['blue', 'green', 'red']

    # Create a histogram plot
    plt.figure(figsize=(10, 6))
    for i in range(num_regions):
        if i == 0:
            label = f'Region {i + 1}'
        else:
            label = f'Region {i + 1}'
        if i == num_regions - 1:
            plt.hist(all_weights[i * elements_per_region:], bins=40, alpha=0.6, color=colors[i], label=label)
        else:
            if i == 1:
                plt.hist(all_weights[i * elements_per_region:(i + 1) * elements_per_region], bins=11, alpha=0.6, color=colors[i], label=label)
            else:
                plt.hist(all_weights[i * elements_per_region:(i + 1) * elements_per_region], bins=40, alpha=0.6, color=colors[i], label=label)


    for boundary in region_boundaries:
        plt.axvline(x=boundary, color='black', linestyle='--')

    
    plt.xlabel('Data Values')
    plt.ylabel('Frequency')


    plt.savefig(plot_name, dpi=300) 

def plot_weight_histogram_three_regions_ternary(model, plot_name):
    all_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            all_weights.extend(weights[0].flatten())  # Weights[0] contains the weight values

    num_regions = 3
    elements_per_region = len(all_weights) // num_regions


    # print('elements per region: {}'.format(elements_per_region))

    
    all_weights.sort()
    region_boundaries = [0] * (num_regions - 1)

    for i in range(1, num_regions):
        region_boundaries[i - 1] = all_weights[i * elements_per_region]


    colors = ['blue', 'green', 'red']

    # Create a histogram plot
    plt.figure(figsize=(10, 6))
    for i in range(num_regions):
        if i == 0:
            label = f'Region {i + 1}'
        else:
            label = f'Region {i + 1}'
        if i == num_regions - 1:
            plt.hist(all_weights[i * elements_per_region:], bins=20, alpha=0.6, color=colors[i], label=label)
        else:
            if i == 1:
                plt.hist(all_weights[i * elements_per_region:(i + 1) * elements_per_region], bins=20, alpha=0.6, color=colors[i], label=label)
            else:
                plt.hist(all_weights[i * elements_per_region:(i + 1) * elements_per_region], bins=20, alpha=0.6, color=colors[i], label=label)


    for boundary in region_boundaries:
        plt.axvline(x=boundary, color='black', linestyle='--')

    
    plt.xlabel('Data Values')
    plt.ylabel('Frequency')



    plt.savefig(plot_name, dpi=300) 


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
print(y_test[3])
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

comms_round = 200
# comms_round = 2


loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )       


build_shape = 784 #(28, 28, 3)  # 1024 <- CIFAR-10    # 784 # for MNIST




actions = [0, 0.25, 0.5, 1]



for run in runs:





    smlp_fp = SimpleMLP()
    

    
    fp_model = smlp_fp.build(build_shape, 10) 
    fp_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)
    
    
    smlp_ternary = SimpleMLP()
    ternary_model = smlp_ternary.build(build_shape, 10) 

    fp_acc = []
    fp_loss = []
    ternary_acc = []
    ternary_loss = []

    for comm_round in range(comms_round):


        # plot_weight_histogram(global_model, 'plots/weights' + str(comm_round) + '.png')



        #randomize client data - using keys
        all_client_names = list(clients_batched.keys())



        # print(all_client_names_rest)   
        client_names = random.sample(all_client_names, k=1)
        client = client_names[0]
        fp_model.fit(clients_batched[client], epochs=1, verbose=0)


        plot_weight_histogram_three_regions(fp_model,'plots/fp_ternary/fp_weights.png')

        print('full-precision:')
        for(X_test, Y_test) in test_batched:
            fp_acc_temp, fp_loss_temp = test_model(X_test, Y_test, fp_model, comm_round)

        fp_acc.append(fp_acc_temp)
        fp_loss.append(fp_loss_temp.numpy().tolist())

        
        ternary_model.set_weights(fp_model.get_weights())
        ternarize_weights(ternary_model)

        print('ternary:')
        for(X_test, Y_test) in test_batched:
            ternary_acc_temp, ternary_loss_temp = test_model(X_test, Y_test, ternary_model, comm_round)
        
        plot_weight_histogram_three_regions_ternary(ternary_model,'plots/fp_ternary/ternary_weights.png')

        ternary_acc.append(ternary_acc_temp)
        ternary_loss.append(ternary_loss_temp.numpy().tolist())


    with open(result_folder_name + 'fp_acc' +'.json', 'w') as file:
        json.dump(fp_acc, file)

    
    with open(result_folder_name + 'fp_loss' +'.json', 'w') as file:
        json.dump(fp_loss, file)


    with open(result_folder_name + 'ternary_acc' +'.json', 'w') as file:
        json.dump(ternary_acc, file)

    
    with open(result_folder_name + 'ternary_loss' +'.json', 'w') as file:
        json.dump(ternary_loss, file)





    
            
            





        
        

