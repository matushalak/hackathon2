import random
import os
import threading
import numpy as np
from testing import predict_single_cfm
from cfms import cfm_main
from unic_data_pull import data_retrieval
from overlap import overlap_main

master_path = os.path.dirname(__file__)
random.seed(42)

# Global variables
continue_running = True
run_count = 1
stored_values = []  # To store (valence, arousal) pairs
valence = None  # Initialize global variables for valence and arousal
arousal = None

def input_listener():
    global continue_running, stored_values, valence, arousal
    while continue_running:
        user_input = input()  # Wait for user input
        if user_input == "":  # Check if Enter was pressed
            if run_count > 1:  # Ensure that valence and arousal have been set
                # Store the current valence and arousal values
                stored_values.append((valence, arousal))
                print(f"Stored valence and arousal: {valence}, {arousal}")

def main():
    global continue_running, run_count, valence, arousal  # Declare global variables to be used

    # Start the input listener thread
    listener_thread = threading.Thread(target=input_listener)
    listener_thread.start()

    while continue_running:
        if run_count == 1:
            raw_data = data_retrieval(8)
            cfm_matrix = cfm_main(raw_data)
            valence, arousal = predict_single_cfm(cfm_matrix, os.path.join(master_path, 'weights.pt'))
            print(f'Current run: {run_count}')
            print(valence, arousal)
            run_count += 1
        else:
            raw_data = overlap_main(raw_data)
            cfm_matrix = cfm_main(raw_data)
            valence, arousal = predict_single_cfm(cfm_matrix, os.path.join(master_path, 'weights.pt'))
            print(f'Current run: {run_count}')
            print(valence, arousal)
            run_count += 1

        # Write the current valence and arousal values to a text file
        np.savetxt(os.path.join(master_path, 'val_ar.txt'), [[valence, arousal]], fmt='%.2f', header='Valence, Arousal')

    print("Data pulling stopped.")
    print("Stored values:", stored_values)  # Print the stored values at the end

if __name__ == "__main__":
    main()
