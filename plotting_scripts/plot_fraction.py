import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Function to read data from a JSON file
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

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def read_result_folder(file_prefix, runs):

    acc_list = list()
    loss_list = list()
    
    file_affix = '_run1.json'  

    for run in runs:
        data = read_json_file(file_prefix + run + file_affix)
        acc, loss = zip(*data)
        acc_list.append(acc)
        loss_list.append(loss)


    return acc_list, loss_list


def add_fractions_to_plot(data_list, fractions):
    nums = range(1,101)
    x=list(nums)
    
    for i in range(len(fractions)):
        label = str(float(fractions[i])*100) + '% Ternary'
        plt.plot(x, data_list[i], label =  label)
        # plt.plot(x, data_list[max_index], color=color, label=label, linewidth=1.5)

dataset_fmnist = 'fmnist'

file_prefix = '../results/prelim_results/fraction/fmnist/global_acc_loss_'
fractions = ['0', '0.2', '0.4', '0.6', '0.8', '1']
fmnist_acc, fmnist_loss = read_result_folder(file_prefix, fractions)


font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
plt.rc('font', **font)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.figure(figsize=(6, 5))

add_fractions_to_plot(fmnist_acc, fractions)

plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
# plt.title(" Fmnist, Training Using IID Data")
plt.legend() 
plt.xlabel("Communication Round")
plt.ylabel('Accuracy')
plt.savefig('../plots/fractions/fmnist.png', dpi=300)
plt.clf()





file_prefix = '../results/prelim_results/fraction/mnist/global_acc_loss_'
fractions = ['0', '0.2', '0.4', '0.6', '0.8', '1']
mnist_acc, mnist_loss = read_result_folder(file_prefix, fractions)


font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
plt.rc('font', **font)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.figure(figsize=(6, 5))

add_fractions_to_plot(mnist_acc, fractions)

plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
# plt.title("mnist, Training Using IID Data")
plt.legend() 
plt.xlabel("Communication Round")
plt.ylabel('Accuracy')
plt.savefig('../plots/fractions/mnist.png', dpi=300)
plt.clf()
