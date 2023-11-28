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


def read_avg_reward_folder(file_prefix, runs):
    max_accs = list()
    file_affix = '_run1.json'

    for episode in episodes:
        max_accs.append(read_json_file(file_prefix + episode + file_affix))

    return max_accs

def add_list_to_plot(data, label, color):
    nums = range(0,25)
    x=list(nums)


    x_plot = [number * 10 for number in x]
    plt.plot(x_plot, data, color=color, label=label, linewidth=2)
  


dataset = 'fmnist'
file_prefix = '../results/average_rewards/' + dataset + '/max_acc_' 
episodes = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240']
max_accs_fmnist = read_avg_reward_folder(file_prefix, episodes)


dataset = 'mnist'
file_prefix = '../results/average_rewards/' + dataset + '/max_acc_' 
episodes = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', '240']
max_accs_mnist = read_avg_reward_folder(file_prefix, episodes)


font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
plt.rc('font', **font)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.figure(figsize=(6, 5))


add_list_to_plot(max_accs_fmnist, 'FMNIST Average Rewards', 'chocolate')
add_list_to_plot(max_accs_mnist, 'MNIST Average Rewards', 'lightseagreen')

plt.xlim([-1, 260])
plt.grid( linestyle = '--', linewidth = 0.5)
plt.legend() 
plt.xlabel("Episode")
plt.ylabel('Reward')
plt.savefig('../plots/average_reward/average_reward' + '.png', dpi=300)
plt.clf()

