import json
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def min_list(list_of_lists):
    min_values = [min(item) for item in zip(*list_of_lists)]
    return min_values

def max_list(list_of_lists):
    max_values = [max(item) for item in zip(*list_of_lists)]
    return max_values


def calculate_average_lists(lists):
    if not lists:
        return []

    num_lists = len(lists)
    list_length = len(lists[0])  # Assuming all lists have the same length

    averages = []

    for i in range(list_length):
        total = sum(lst[i] for lst in lists)
        average = total / num_lists
        averages.append(average)

    return averages

def calculate_average(lst):
    if not lst:  # Check if the list is empty to avoid division by zero
        return 0

    # Calculate the sum of all elements in the list
    total_sum = sum(lst)

    # Calculate the average by dividing the sum by the number of elements in the list
    average = total_sum / len(lst)

    return average

def read_csv_first_column(file_path):
    first_column = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row) > 0:
                first_column.append(float(row[0]))
    return first_column

def acc_difference(client_acc, prev_global_acc):
    
    # Ensure both lists have the same length
    if len(client_acc) != len(prev_global_acc):
        raise ValueError("Both lists must have the same length.")

    # Calculate the absolute difference between corresponding elements and sum them up
    return sum((a - b) for a, b in zip(client_acc, prev_global_acc))


def subtract_lists(list1, list2):
    result = [x - y for x, y in zip(list1, list2)]
    return result

def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def decompose_acc_loss_tuple(input):
    acc = list()
    loss = list()


    for i in range(len(input)):
        temp_acc, temp_loss = input[i]
        acc.append(temp_acc)
        loss.append(temp_loss)

    return acc, loss   

def decompose_acc_loss_action_tuple(input):
    acc = list()
    loss = list()
    action = list()

    for i in range(len(input)):
        temp_acc, temp_loss, temp_action = input[i]
        acc.append(temp_acc)
        loss.append(temp_loss)
        action.append(temp_action)

    return acc, loss, action

def plot_actions(RL_actions):
    x_values = range(len(RL_actions))
    plt.bar(x, RL_actions)
    plt.xlim([0, 101])
    plt.ylim([0, 1])

    

    plt.xlabel("Communication Round")
    plt.ylabel("Action")

    plt.grid( linestyle = '--', linewidth = 0.5)
    
    plt.title("Actions in Each Comm. Round (Run2)")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=25))  # Set interval to 1

    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(base=0.25))  # Set interval to 5

    # plt.text(90, 1.2, 'Average of actions: ' + str("{:.2f}".format(sum(RL_actions)/len(RL_actions))), style='italic', bbox={'facecolor': 'wheat', 'alpha': 0.5, 'pad': 10})

    plt.savefig('actions.png', dpi=300)
    plt.clf()
    

def plot_faded_list(x, RL_accs, FL_accs, dataset_name, metric):
    plt.plot(x, calculate_average_lists(RL_accs), color='green', label='RLAvg')
    plt.fill_between(x, min_list(RL_accs), max_list(RL_accs), color='green',alpha=0.25, linewidth=0)

    plt.plot(x, calculate_average_lists(FL_accs), color='royalblue', label='FedAvg')
    plt.fill_between(x, min_list(FL_accs), max_list(FL_accs), color='royalblue',alpha=0.25, linewidth=0)

    # plt.plot(x, pweighted_acc, color='sienna', label='Precision-WeightedAvg')

    plt.xlim([0, 101])

    # plt.ylim([0.2, 1])

    plt.grid( linestyle = '--', linewidth = 0.5)

    plt.title(dataset_name + " Training Using Non-IID Data")
    plt.legend() 

    plt.xlabel("Communication Round")
    plt.ylabel(metric)

    plt.savefig('RLvsFL.png', dpi=300)
    plt.clf()


folder_names = ['categorical_RL/', 'random_action_IID_4/', 'compare/']
file_names = ['accs_all_clients_', 'global_accs']

nums = range(1,101)
x=list(nums)


RL_acc = list()
RL_loss = list()
RL_action = list()



for i in range(5):
    acc = list()
    loss = list()
    action = list()
    
    input_list = read_json('results/fmnist_Non_IID_0.5/global_acc_loss_action_run' + str(i+1) + '_40' + '.json')
    acc, loss, action = decompose_acc_loss_action_tuple(input_list)
    RL_acc.append(acc)
    RL_loss.append(loss)
    RL_action.append(action)

FL_accs = list()
FL_loss = list()

for i in range(5):
    acc = list()
    loss = list()
    
    input_list = read_json('results/benchmarks/tmean/fmnist/NIID(done)/global_acc_loss_action_run' + str(i+1) + '.json')
    acc, loss = decompose_acc_loss_tuple(input_list)
    FL_accs.append(acc)
    FL_loss.append(loss)




# print(RL_acc)
# print(RL_action)


plot_faded_list(x, RL_loss, FL_loss, 'FMNIST', 'Loss')

# plot_actions(RL_action[0])


