import json
import matplotlib.pyplot as plt

# Function to read data from a JSON file
def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# File paths for your JSON files
file1_path = '../results/prelim_results/fp_ternary/fp_acc.json'
file2_path = '../results/prelim_results/fp_ternary/ternary_acc.json'

# Read data from the JSON files
data1 = read_json_file(file1_path)
data2 = read_json_file(file2_path)

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}

# Set the font properties
plt.rc('font', **font)
plt.rc('axes', titlesize=32)
plt.rc('axes', labelsize=28)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=24)

# Create a plot using Matplotlib
plt.figure(figsize=(10, 8))  # Set the figure size

# Plot data from the first file in blue
plt.plot(data1, label='Full Precision', color='seagreen', linewidth = 3)

# Plot data from the second file in red
plt.plot(data2, label='Ternary', color='royalblue' , linewidth = 3)

plt.grid( linestyle = '--', linewidth = 0.5)

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.savefig('../plots/fp_ternary/fp_ternary_acc.png', dpi=300)
plt.clf()


# File paths for your JSON files
file1_path = '../results/prelim_results/fp_ternary/fp_loss.json'
file2_path = '../results/prelim_results/fp_ternary/ternary_loss.json'

# Read data from the JSON files
data1 = read_json_file(file1_path)
data2 = read_json_file(file2_path)

# Create a plot using Matplotlib
plt.figure(figsize=(10, 8))  # Set the figure size

# Plot data from the first file in blue
plt.plot(data1, label='Full Precision', color='seagreen', linewidth = 3)

# Plot data from the second file in red
plt.plot(data2, label='Ternary', color='royalblue', linewidth = 3)

plt.grid( linestyle = '--', linewidth = 0.5)

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the plot
plt.savefig('../plots/fp_ternary/fp_ternary_loss.png', dpi=300)
plt.clf()

