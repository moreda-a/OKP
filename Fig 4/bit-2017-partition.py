import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv("BTC-2017min.csv")

# Get the number of rows in the DataFrame
num_rows = len(df)

# Specify the number of partitions
s = 12

# Specify the number of random numbers you want to select for each partition
n = 10000  # Change this to the desired number

# Loop to create 's' number of files
for i in range(1, s + 1):
    # Calculate the range for the partition
    start_index = int((i - 1) / s * (num_rows - 1)) + 1  # Exclude the first row
    end_index = int(i / s * (num_rows - 1)) + 1  # Exclude the first row
    
    # Select n random rows from the current partition
    random_rows = random.sample(range(start_index, end_index), n)
    random_rows.sort()  # Sort the selected rows
    
    # Create a new file with the specified name
    filename = f"con_numbers_{i}.txt"
    with open(filename, "w") as file:
        # Write each randomly selected rounded number to the file
        for row_index in random_rows:
            rounded_number = round(df.iloc[row_index, 3])  # 4th column index is 3 (0-based)
            file.write(str(rounded_number) + "\n")
    
    print("Random numbers from partition", i, "have been written to", filename)