import pickle

# Initialize an empty list to store the combined data
combined_data = []

# List of pickle file paths
pickle_files = [
                # "results/constant/fmnist_Non_IID_0.5/data.pkl", 
                # "results/constant/fmnist_Non_IID_0.25/data.pkl", 
                # "results/constant/fmnist_Non_IID_1/data.pkl",
                "results/start_from/fmnist_Non_IID_0.5/data.pkl", 
                "results/start_from/fmnist_Non_IID_0.25/data.pkl", 
                "results/start_from/fmnist_Non_IID_1/data.pkl",
                # "results/constant/fmnist_Non_IID_0/data.pkl"
                # "results/random_action/mnist_Non_IID/data.pkl"
                ]

# Iterate through each pickle file
for file_path in pickle_files:
    try:
        # Load the data from the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Append the data to the combined_data list
        # print(file_path)
        # print(data[0])
        # print(data[1])
        combined_data.extend(data)
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.UnpicklingError:
        print(f"Error unpickling data from {file_path}")

# Now, combined_data contains the combined list of tuples from all the pickle files
print(len(combined_data))
# If you want to save the combined data to a new pickle file
with open("results/merged_data/fmnist.pkl", 'wb') as combined_file:
    pickle.dump(combined_data, combined_file)

# Optionally, you can also print or work with the combined_data as needed.
