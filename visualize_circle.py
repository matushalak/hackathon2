import matplotlib.pyplot as plt
import numpy as np
import time

while True:
    start = time.time()
    valence, arousal = np.random.randint(1,5,2)
    valences = np.random.normal(valence, 0.25, 10)
    arousals = np.random.normal(arousal, 0.25, 10)

    while time.time()-start < 4:
        for v, a, in zip(valences, arousals):
            plt.plot(v, a, 'ro')
            plt.xlim(0,6)
            plt.ylim(0,6)
            plt.axis('off')
            plt.pause(4 / len(valences))
            plt.cla()
        pass
    continue


