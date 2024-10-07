import os
import numpy as np
from unic_data_pull import data_retrieval

def overlap_main(old_data):
    old_data_half = old_data[:, 1000:]
    new_data_half = data_retrieval(4) # run for 4 seconds to have the overlap
    new_raw_data = np.concatenate((old_data_half, new_data_half), axis=1)    

    return new_raw_data 

if __name__ == "__main__":
    overlap_main()