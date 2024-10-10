import pandas as pd
import numpy as np
import os
import random

# Read all 86 files and combine into one DataFrame
combined_values = []
combined_weights = []

for i in range(1, 87):  # Assuming filenames are from 1 to 86
    filename = f'values250/jobvalue{i}.csv'
    if os.path.exists(filename):
        data = pd.read_csv(filename, header=None)
        combined_values.extend(data[0].values)  # First column is jobValue
        combined_weights.extend(data[1].values)  # Second column is jobWeight
    else:
        print(f"File {filename} not found.")

# Add 70 to all combined values
combined_values = [value + 70 for value in combined_values]

# Report minimum and maximum values
min_value = min(combined_values)
max_value = max(combined_values)
print(f"Minimum Value after adding 70: {min_value}")
print(f"Maximum Value after adding 70: {max_value}")

# Shuffle values and weights
random.shuffle(combined_values)
random.shuffle(combined_weights)

# Total number of values
total_values = len(combined_values)

# Set the total number of lines for each file and number of files to create
lines_per_file = 10000
num_files = 24

# Function to get value based on mod logic
def get_mod_value(index, combined_list):
    return combined_list[index % total_values]

# Create 24 files, each with 10,000 lines of values
for i in range(num_files):
    output_filename = f'con_numbers_{i+1}.txt'
    with open(output_filename, 'w') as f:
        for j in range(lines_per_file):
            # Get the value using the modulo if the index exceeds the available number of values
            value = get_mod_value(i * lines_per_file + j, combined_values)
            f.write(f"{value}\n")

print("24 files created with shuffled values (70 added to all values).")