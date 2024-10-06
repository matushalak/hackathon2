import random
import os
from testing import predict_single_cfm
from cfms import cfm_main

random.seed(42)
master_path = os.path.dirname(__file__)

def main():
    cfm_matrix = cfm_main('raw_data1')
    valence, arousal = predict_single_cfm(cfm_matrix, os.path.join(master_path, 'weights.pt'))
    print(valence, arousal)

if __name__ == "__main__":
    main()