import os
import json
import pickle

#make state/action/reward/next_state

round = 100
no_clients = 10
max_loss = 4
# Define the folder path where your JSON files are located
folder_path = 'results/start_from/fmnist_Non_IID_1'

# parsed data
accs_clients_data = {}
global_acc_loss_action_data = {}
global_accs_data = {}
max_acc_data = {}

#generated data
actions_data = {}
loss_data = {}
acc_data = {}
state = {}
next_state = {}


# Create a list of file patterns to look for
file_patterns = [
    'accs_clients_run',
    'global_acc_loss_action_run',
    'global_accs_run',
    'max_acc_run'
]

action_dict = {0:0, 0.25:1, 0.5: 2, 1:3}

# Iterate through the files in the folder
for root, dirs, files in os.walk(folder_path):
    for file_name in files:

        # Check if the file matches any of the specified patterns
        for pattern in file_patterns:
            if pattern in file_name:
                # Extract the run number from the file name
                run_number = file_name.split(pattern)[-1].strip('.json')
                
                # Create the full file path
                file_path = os.path.join(root, file_name)
                
                # Read the JSON data from the file
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                
                # Store the JSON data in the appropriate dictionary based on the pattern
                if pattern == 'accs_clients_run':
                    accs_clients_data[f'run{run_number}'] = json_data
                elif pattern == 'global_acc_loss_action_run':
                    global_acc_loss_action_data[f'run{run_number}'] = json_data
                elif pattern == 'global_accs_run':
                    global_accs_data[f'run{run_number}'] = json_data
                elif pattern == 'max_acc_run':
                    max_acc_data[f'run{run_number}'] = json_data



for key in global_acc_loss_action_data.keys():

    actions_data[key] = [sublist[-1] for sublist in global_acc_loss_action_data[key]] 
    loss_data[key] = [sublist[-2] for sublist in global_acc_loss_action_data[key]] 
    acc_data[key] = [sublist[0] for sublist in global_acc_loss_action_data[key]]



for key in global_acc_loss_action_data.keys():

    temp_list_of_states = list()

    initial_state = [0] * 100 + [0] * 10 + [0] + [1] + [0]
    temp_list_of_states.append(initial_state)

    for i in range(round):
        temp_state = list()
        for j in range(no_clients):
            temp_state = temp_state + accs_clients_data[key][j][i]
        
        temp_state = temp_state + global_accs_data[key][i]
        temp_state = temp_state + [acc_data[key][i]]
        temp_state = temp_state + [loss_data[key][i]/max_loss]
        temp_state = temp_state + [(i+1)/100]
        
        
        temp_list_of_states.append(temp_state)

    state[key] = temp_list_of_states


for key in global_acc_loss_action_data.keys():
    temp_list_of_next_states = list()

    temp_list_of_next_states = state[key][1:]
    next_state[key] = temp_list_of_next_states



for key in global_acc_loss_action_data.keys():
    state[key].pop()




#construct the tuple for each run







#construct the tuple list
list_of_tuples = list()

for key in state.keys():
    print(key)
    for i in range(round):
        list_of_tuples.append((state[key][i], action_dict[actions_data[key][i]], max_acc_data[key], next_state[key][i]))



print(len(list_of_tuples))

file_path = folder_path + '/data.pkl'

# Open the file in binary write mode
with open(file_path, 'wb') as file:
    # Serialize and dump the list of tuples into the file
    pickle.dump(list_of_tuples, file)

print(f"List of tuples has been dumped into {file_path}")





