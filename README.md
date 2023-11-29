# RLFL: A Reinforcement Learning-Based Aggregation Approach for Federated Learning in Full Precision and Ternary Environments

## Overview

This repository contains the implementation of a Reinforcement Learning (RL) controller for Federated Learning (FL) in Full Precision and Ternary Environments. 

Directories:
```

RLFL/
├── benchmarks/
├── keras_mnist/
├── keras_fmnist/
├── plots/
├── plotting_scripts/
├── results/
├── saved_RL_agents/
... (other execution scripts and parameter text files)
└── README.md
```



(main scripts and files holding parameters)


## Requirements
All the required dependencies are listed in requirements.txt and the important ones are listed below:
- Python 3.8.18
- keras 2.8.0
- tensorflow 2.8.0
- scikit-learn 0.24.2

## Running the Trained Models

For running the trained RL models go to the root directory and execute the following command for MNIST or FEMNIST:
python -W ignore test_RL_model_mnist.py
python -W ignore test_RL_model_fmnist.py

## RL Training
To continue the training process of the RL agent run the following commands edit the runs parameter at the start of the following codes and run them as below:
python -W ignore RL_mnist_training_continue.py
python -W ignore RL_mnist_training_continue.py

if you want to start the training from the beginning for MNIST for example delete the files alpha_mnist, episode_mnist, epsilon_mnist and replay_mnist and edit the runs parameter in the code to start from 'run0' and then rerun the codes.

## Running Benchmarks
To run the fedasl benchmark for example edit the runs parameter and dataset parameter in the code and run the benchmark with the following commands:
cd benchmarks
python -W ignore fedasl.py

## Plotting the Results
Use the scripts in the plotting_scripts folder to plot the figures based on the results that are available in the results folder.




