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

def subtract_half(original_list):
    modified_list = [x - 0.5 for x in original_list]
    return modified_list

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





# Convert class vectors to binary class matrices. This is called one hot encoding.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)



build_shape = 784

                                                 


clients = create_clients(X_train, y_train, num_clients=160, initial='client')



clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)






#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))



input_shape_RL = 10 * 10 + 10 # local accs, previous global acc
num_actions_RL = 5 # 0, 0.25, 0.5, 0.75, 1


if(os.path.exists('keras_fmnist_IID')):
    print('loading saved model')
    value_network = tf.keras.models.load_model('keras_fmnist_IID')
else:
    print('creating a new model')
    value_network = tf.keras.models.Sequential([
    tf.keras.layers.Dense(440, activation='relu', input_shape=(input_shape_RL,)),
    tf.keras.layers.Dense(440, activation='relu'),
    tf.keras.layers.Dense(num_actions_RL)
    ])


optimizer_RL = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn_RL = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")



gamma_RL = 0.95
batch_RL = 400

if(os.path.exists('replay_RL_fmnist_IID.pkl')):
    with open('replay_RL_fmnist_IID.pkl', 'rb') as file:
        replay_RL = pickle.load(file)
else:
    replay_RL = deque(maxlen=5000)

if(os.path.exists('epoch_fmnist_IID.txt')):
    with open('epoch_fmnist_IID.txt', 'r') as file:
        epoch_RL = int(file.read())
else: 
    epoch_RL = 0

if(os.path.exists('epsilon_fmnist_IID.txt')):
    with open('epsilon_fmnist_IID.txt', 'r') as file:
        epsilon_RL = float(file.read())
else: 
    epsilon_RL = 1
    
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



runs = [  'run33','run34', 'run35', 'run36', 'run37',
'run38', 'run39','run40','run41','run42', 'run43','run44', 'run45', 'run46', 'run47']
# runs = ['run20','run21','run22', 'run23','run24', 'run25', 'run26', 'run27', 'run28', 'run29']
random_flag = 'random'

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



        # print(all_client_names_rest)   
        client_names = random.sample(all_client_names, k=10)

        # print(client_names, len(client_names))
        random.shuffle(client_names)

                    
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


            client = client_names[iii]
            local_model.fit(clients_batched[client], epochs=1, verbose=0)


            
            


            scaling_factor = 0.1 # weight_scalling_factor(clients_batched, client)

            local_weights = local_model.get_weights()
            

            
            num_ones = 0
            num_mones = 0
            num_zeros = 0
            
            

            if(counter < 4):


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
        
        ############# making the next state and doing batch training
        next_state = []

        if(comm_round>1): 

            for iiii in range(10):
                next_state = next_state + subtract_half(accs_all_clients[iiii][comm_round])
                
            next_state = next_state + subtract_half(global_accs[comm_round - 1])
            next_state = np.array(next_state)

        if(comm_round > 2):
            
            replay_RL.append((state,action,reward,next_state,done))
            with open('replay_RL_fmnist_IID.pkl', 'wb') as file:
                pickle.dump(replay_RL, file)

            
            if len(replay_RL)>batch_RL:
                with tf.GradientTape() as tape:
                    batch_ = random.sample(replay_RL,batch_RL)
                    q_value1 = value_network(tf.convert_to_tensor([x[0] for x in batch_]))
                    q_value2 = value_network(tf.convert_to_tensor([x[3] for x in batch_]))
                    
                    reward = tf.convert_to_tensor([x[2] for x in batch_])
                    action = tf.convert_to_tensor([x[1] for x in batch_])
                    done =   tf.convert_to_tensor([x[4] for x in batch_])
        
                    
                    actual_q_value1 = tf.cast(reward,tf.float64) + tf.cast(tf.constant(alpha_RL),tf.float64)*(tf.cast(tf.constant(gamma_RL),tf.float64)*tf.cast((tf.constant(1)-done),tf.float64)*tf.cast(tf.reduce_max(q_value2),tf.float64))           
                    loss_RL = tf.cast(tf.gather(q_value1,action,axis=1,batch_dims=1),tf.float64)
                    loss_RL = loss_RL - actual_q_value1
                    loss_RL = tf.reduce_mean(tf.math.pow(loss_RL,2))

            
                    grads = tape.gradient(loss_RL, value_network.trainable_variables)
                    optimizer_RL.apply_gradients(zip(grads, value_network.trainable_variables))

                    print('Episode {} Epoch {} done with loss {} !!!!!!'.format(episode_RL, epoch_RL,loss_RL))
                    episode_RL += 1
                    value_network.save('keras_fmnist_IID/')
                    if epoch_RL%100==0:
                        epsilon_RL*=0.997
                    epoch_RL+=1

                    with open('epoch_fmnist_IID.txt', 'w') as file:
                        file.write(str(epoch_RL))

                    with open('epsilon_fmnist_IID.txt', 'w') as file:
                        file.write(str(epsilon_RL))
                    
                    
            
        
        #############
        
        state = next_state
        

        if(comm_round > 2):
            input_data = np.array(state).reshape(-1, *state.shape)
            value_function = value_network.predict(input_data)[0]

            if np.random.rand()>epsilon_RL:
                print("non random action")
                action = np.argmax(value_function)
                random_flag = 'non_random'
            else:
                print("random action")
                action = np.random.choice(num_actions_RL)
                random_flag = 'random'
            action_to_take = actions[action]

        else:
            action = 4
            action_to_take = 1
        
        
        
        
        sum_of_scales = 0.6 + (action_to_take * 0.4)

        
         
        #to get the average over all the local model, we simply take the sum of the scaled weights
        for iiii in range(4):
            # print('before:', scaled_local_weight_list[iiii])
            scaled_local_weight_list[iiii] = scale_model_weights(scaled_local_weight_list[iiii], action_to_take)
            # print('after:', scaled_local_weight_list[iiii])

        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        average_weights = scale_model_weights(average_weights, 1/sum_of_scales)

        #update global model 
        global_model.set_weights(average_weights)

        temp_global_accs = []
        for(X_test, Y_test) in test_batched:
            temp_global_accs = test_model_categorical(X_test, Y_test, local_model)

        global_accs.append(temp_global_accs)

        all_actions.append((action_to_take, random_flag))

        #test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            print(run)
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
        


    
    with open('results/fmnist/RL_IID/accs_all_clients_IID_RL_' + run + '.json', 'w') as file:
        json.dump(accs_all_clients, file)

    with open('results/fmnist/RL_IID/global_accs_IID_RL_' + run + '.json', 'w') as file:
        json.dump(global_accs, file)

    with open('results/fmnist/RL_IID/actions_RL_' + run + '.json', 'w') as file:
        json.dump(all_actions, file)

    iid_df = pd.DataFrame(list(zip(global_acc_list, global_loss_list)), columns=['global_acc_list', 'global_loss_list'])
    iid_df.to_csv('results/fmnist/RL_IID/MNIST_IID_RL_'  + run + '.csv',index=False)


