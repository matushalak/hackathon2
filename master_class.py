import random
import os
import threading
from testing import predict_single_cfm
from cfms import cfm_main
from unic_data_pull import data_retrieval
from overlap import overlap_main
import numpy as np

class DataProcessor():
    def __init__(self):
        self.master_path = os.path.dirname(__file__)
        random.seed(42)

        # Global variables
        self.continue_running = True
        self.run_count = 1
        self.stored_values = []  # To store (valence, arousal) pairs
        self.valence = None  # Initialize global variables for valence and arousal
        self.arousal = None

    def input_listener(self):
        """Listen for the Enter key press to store current values."""
        while self.continue_running:
            user_input = input()  # Wait for user input
            if user_input == "":  # Check if Enter was pressed
                if self.run_count > 1:  # Ensure that valence and arousal have been set
                    # Store the current valence and arousal values
                    self.stored_values.append((self.valence, self.arousal))
                    print(f"Stored valence and arousal: {self.valence}, {self.arousal}")

    def start_input_listener(self):
        """Start the input listener in a separate thread."""
        listener_thread = threading.Thread(target=self.input_listener)
        listener_thread.start()

    def run(self):
        """Main processing loop."""
        self.start_input_listener()  # Start the input listener

        while self.continue_running:
            if self.run_count == 1:
                raw_data = data_retrieval(8)
                cfm_matrix = cfm_main(raw_data)
                self.valence, self.arousal = predict_single_cfm(cfm_matrix, os.path.join(self.master_path, 'weights.pt'))
                np.savetxt('val_ar.txt',[self.valence,self.arousal])
                print(f'Current run: {self.run_count}')
                print(self.valence, self.arousal)
                self.run_count += 1
            else:
                raw_data = overlap_main(raw_data)
                cfm_matrix = cfm_main(raw_data)
                self.valence, self.arousal = predict_single_cfm(cfm_matrix, os.path.join(self.master_path, 'weights.pt'))
                np.savetxt('val_ar.txt',[self.valence,self.arousal])
                print(f'Current run: {self.run_count}')
                print(self.valence, self.arousal)
                self.run_count += 1

        print("Data pulling stopped.")
        print("Stored values:", self.stored_values)  # Print the stored values at the end

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()
