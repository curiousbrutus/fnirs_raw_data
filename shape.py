import numpy as np

# Load the .npy file
data = np.load('/home/jobbe/Desktop/Thesis_Mind-fMRI/pps_fnirs/sub_70_preprocessed.npy')

# Now you can use the 'data' variable to access the contents of the .npy file
# For example, if it contains a numpy array, you can print its shape
print("Shape of the array:", data.shape)

# Or you can print the array itself
print("Contents of the array:")
print(data)

