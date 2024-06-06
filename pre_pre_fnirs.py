import os
import numpy as np
from scipy import interpolate

def pad_to_length(x, length):
    assert x.ndim == 2
    assert x.shape[-1] <= length
    if x.shape[-1] == length: 
        return x
    return np.pad(x, ((0, 0), (0, length - x.shape[1])), mode='constant')

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    num_to_generate = int((aug_times - 1) * len(data))
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z, axis=0))
    data_aug = np.concatenate(data_aug, axis=0)
    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    values = np.stack((x, y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1, 1)]
    z = interpolate.interpn(points, values, xi)
    return z

def preprocess_fnirs_data(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .csv files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in files:
        # Load data from the .csv file
        data = np.loadtxt(os.path.join(input_dir, file), delimiter=',', skiprows=1)

        # Normalize the data
        data = (data - np.mean(data)) / np.std(data)

        # Perform padding to a fixed length
        padded_data = pad_to_length(data, desired_length)

        # Perform data augmentation
        augmented_data = augmentation(padded_data, aug_times=2)

        # Save preprocessed data to .npy files
        np.save(os.path.join(output_dir, file[:-4] + '_preprocessed.npy'), augmented_data)

    print("Preprocessing completed.")

# Set input and output directories
input_dir = '/home/jobbe/Desktop/Thesis_Mind-fMRI/whole_data_fnirs'
output_dir = '/home/jobbe/Desktop/Thesis_Mind-fMRI/pps_fnirs'
desired_length = 1000  # Set the desired length for padding

# Preprocess the fNIRS data
preprocess_fnirs_data(input_dir, output_dir)

