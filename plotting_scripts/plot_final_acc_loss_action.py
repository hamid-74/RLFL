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


def read_RLFL_folder(file_prefix, runs):
    accs_RLFL = list()
    loss_RLFL = list()
    actions_RLFL = list()

    
    file_affix = '.json'
    

    for run in runs:
        data = read_json_file(file_prefix + run + file_affix)
        acc, loss, action = zip(*data)
        accs_RLFL.append(acc)
        loss_RLFL.append(loss)
        actions_RLFL.append(action)

    return accs_RLFL, loss_RLFL, actions_RLFL

def read_result_folder(file_prefix, runs):

    acc_list = list()
    loss_list = list()
    
    file_affix = '.json'  

    for run in runs:
        data = read_json_file(file_prefix + run + file_affix)
        acc, loss = zip(*data)
        acc_list.append(acc)
        loss_list.append(loss)


    return acc_list, loss_list

def add_RLFL_max_action_to_plot(data_list, max_index, label, color):

    nums = range(1,101)
    x=list(nums)
    # plt.bar(x, data_list[max_index], color=color, linewidth=1.5)
    
    plt.bar(x, data_list[max_index], color='palegreen')
    plt.scatter(x, data_list[max_index], marker='o', linestyle='-', color='green')
    
def add_RLFL_max_fade_to_plot(data_list, label, color):

    max_index = max(range(len(data_list)), key=lambda i: max(data_list[i]))

    nums = range(1,101)
    x=list(nums)
    plt.plot(x, data_list[max_index], color=color, label=label, linewidth=1.5)
    plt.fill_between(x, min_list(data_list), max_list(data_list), color=color, alpha=0.25, linewidth=0)

    return max_index


def add_RLFL_average_to_plot(data_list, label, color):

    nums = range(1,101)
    x=list(nums)
    plt.plot(x, calculate_average_lists(data_list), color=color, label=label, linewidth=3)



def add_average_to_plot(data_list, label, color):

    nums = range(1,101)
    x=list(nums)

    plt.plot(x, calculate_average_lists(data_list), color=color, label=label, linewidth=2)




def add_RLFL_average_to_plot_network_overhead(data_list, scalar, label, color):

    nums = range(1,101)
    x=list(nums)
    x_plot = [number * scalar for number in x]
    plt.plot(x_plot, calculate_average_lists(data_list), color=color, label=label, linewidth=3)
  

def add_average_to_plot_network_overhead(data_list, scalar,label, color):

    nums = range(1,101)
    x=list(nums)
    x_plot = [number * scalar for number in x]
    plt.plot(x_plot, calculate_average_lists(data_list), color=color, label=label,linewidth=2)
    


dataset = 'mnist'

file_prefix = '../results/RL_test_final/' + dataset + '_NIID/global_acc_loss_action_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
RLFL_acc, RLFL_loss, RLFL_actions = read_RLFL_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/fedasl/' + dataset + '/NIID(done)/global_acc_loss_action_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
fedasl_acc, fedasl_loss = read_result_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/median/' + dataset + '/NIID(done)/global_acc_loss_action_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
median_acc, median_loss = read_result_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/tmean/' + dataset + '/NIID(done)/global_acc_loss_action_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
tmean_acc, tmean_loss = read_result_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/t_fedavg/' + dataset + '/NIID/global_acc_loss_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
t_fedavg_acc, t_fedavg_loss = read_result_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/fedavg_h/' + dataset + '/NIID/global_acc_loss_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
fedavg_h_acc, fedavg_h_loss = read_result_folder(file_prefix, runs)

file_prefix = '../results/benchmarks/fl/' + dataset + '/NIID/global_acc_loss_'
runs = ['run1', 'run2', 'run3', 'run4', 'run5']
fl_acc, fl_loss = read_result_folder(file_prefix, runs)



font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
plt.rc('font', **font)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.figure(figsize=(6, 5))





### Average acc comparison
add_RLFL_average_to_plot(RLFL_acc, 'RLFL', 'seagreen')
add_average_to_plot(fedasl_acc, 'FedAsl', 'royalblue')
add_average_to_plot(median_acc, 'Median', 'darkgoldenrod')
add_average_to_plot(tmean_acc, 'Trimmed Mean', 'purple')


plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
plt.title( dataset + " Dataset, Training Using Non-IID Data")
plt.legend() 
plt.xlabel("Communication Round")
plt.ylabel('Accuracy')
plt.savefig('../plots/final_comparison/acc_comparison_' + dataset + '.png', dpi=300)
plt.clf()

### Average loss comparison
add_RLFL_average_to_plot(RLFL_loss, 'RLFL', 'seagreen')
add_average_to_plot(fedasl_loss, 'FedAsl', 'royalblue')
add_average_to_plot(median_loss, 'Median', 'darkgoldenrod')
add_average_to_plot(tmean_loss, 'Trimmed Mean', 'purple')

plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
plt.title( dataset + " Dataset, Training Using Non-IID Data")
plt.legend() 
plt.xlabel("Communication Round")
plt.ylabel('Loss')
plt.savefig('../plots/final_comparison/loss_comparison_' + dataset + '.png', dpi=300)
plt.clf()


### RLFL best performance
max_index = add_RLFL_max_fade_to_plot(RLFL_acc, 'RLFL', 'seagreen')

plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
plt.title( dataset + " Dataset, Training Using Non-IID Data")
plt.legend() 
plt.xlabel("Communication Round")
plt.ylabel('Accuracy')
plt.savefig('../plots/final_comparison/RLFL_bestperformance_' + dataset + '.png', dpi=300)
plt.clf()


### Action plot
add_RLFL_max_action_to_plot(RLFL_actions, max_index,'RLFL', 'seagreen')

plt.xlim([0, 101])
plt.ylim([-0.25, 1.25])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title( dataset + " Dataset, Training Using Non-IID Data")
plt.xlabel("Communication Round")
plt.ylabel('Action')
plt.savefig('../plots/final_comparison/RLFL_bestperformance_actions_' + dataset + '.png', dpi=300)
plt.clf()




### Average acc comparison with network overhead
scalar_RLFL = (198800 * 2 * 4 + 198800 * 32 * 6)/1000000 #Mbits per_iteration
scalar_t_fedavg = (198800 * 2 * 10)/1000000 #Mbits per_iteration
scalar_FL_H = scalar_RLFL #Mbits per_iteration
scalar_FL = (198800 * 32 * 10)/1000000 #Mbits per_iteration

print('T_FEDAVG: {}, RLFL: {}'.format(scalar_t_fedavg, scalar_RLFL))

add_RLFL_average_to_plot_network_overhead(RLFL_acc, scalar_RLFL/8,'RLFL', 'seagreen')
add_average_to_plot_network_overhead(t_fedavg_acc, scalar_t_fedavg/8,'T-FedAvg', 'royalblue')
add_average_to_plot_network_overhead(fedavg_h_acc, scalar_FL_H/8,'FedAvg-H', 'darkgoldenrod')
add_average_to_plot_network_overhead(fl_acc, scalar_FL/8,'FedAvg', 'purple')

# plt.xlim([0, 101])
plt.grid( linestyle = '--', linewidth = 0.5)
plt.title( dataset + " Dataset, Training Using Non-IID Data")
plt.legend() 
plt.xlabel("Uploaded Data Volume (MB)")
plt.ylabel('Accuracy')
plt.savefig('../plots/final_comparison/acc_comparison_network_overhead_' + dataset + '.png', dpi=300)
plt.clf()


