import os
import numpy as np
from unic_data_pull import data_retrieval

def overlap():
    raw_data = None  # Initialize raw_data

    while True:
        # Retrieve the current raw data
        if raw_data is None:
            raw_data = data_retrieval()  # First retrieval to initialize raw_data
        else:
            # Use the previously modified raw_data for subsequent loops
            raw_data_past = raw_data.T[:, 1000:]  # Get last 1000 columns
            new_data = data_retrieval(save_file=False)  # Retrieve new data
            raw_data[:, :1000] = raw_data_past  # Replace the first 1000 columns with past data
            raw_data[:, 1000:] = new_data[:, :1000]  # Append new data to the end
        
        # Save the modified raw data
        np.save(os.path.join(savepath, f'raw_data{current_csv_num + 1}.npy'), raw_data.T)

        print(f"Iteration {loop + 1}: Data saved as raw_data{current_csv_num + 1}.npy")
        loop += 1
    
    # After every 5 loops, ask the user if they want to continue
    continue_retrieval = input('Do you want to continue retrieving data? (y/n): ').strip().lower()
    if continue_retrieval != 'y':
        print("Stopping data retrieval.")
        break  # Exit the loop
    else:

if __name__ == "__main__":
    overlap()