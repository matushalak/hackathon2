import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Circle

# Calculate the radius
# Set the radius to 2 units
radius = 2.5

# Create the figure and axis outside the loop
fig, ax = plt.subplots()
# Create a circle patch with the specified radius, centered at (3,3)
circle = Circle((3, 3), radius, fill=False, color='black')
# Add the circle to the axis
ax.add_patch(circle)

# Define colors for each quadrant
quadrant_colors = {
    'Q1': 'red',
    'Q2': 'green',
    'Q3': 'blue',
    'Q4': 'purple'
}

while True:
    start = time.time()
    valence, arousal = np.random.randint(1, 5, 2)
    valences = np.random.normal(valence, 0.25, 20)
    arousals = np.random.normal(arousal, 0.25, 20)

    while time.time() - start < 4:
        for v, a in zip(valences, arousals):
            # Clear only the plotted data

            # Determine the quadrant
            if v > 3 and a > 3:
                color = quadrant_colors['Q1']
            elif v < 3 and a > 3:
                color = quadrant_colors['Q2']
            elif v < 3 and a < 3:
                color = quadrant_colors['Q3']
            elif v > 3 and a < 3:
                color = quadrant_colors['Q4']
            else:
                color = 'black'  # If exactly on the axis, color it black

            # Plot the point with the corresponding color
            ax.plot(v, a, 'o', color=color)

            # Set limits and set aspect ratio
            ax.set_xlim(.5, 5.5)
            ax.set_ylim(.5, 5.5)
            ax.set_ylabel('Arousal')
            ax.set_xlabel('Valence')
            ax.set_aspect('equal')

            # Adjust the spines to cross at (3,3)
            ax.spines['left'].set_position(('data', 3))
            ax.spines['bottom'].set_position(('data', 3))
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Set ticks position
            # ax.xaxis.set_ticks_position('bottom')
            # ax.yaxis.set_ticks_position('left')

            # # Set tick labels
            # ax.set_xticks(np.arange(.5, 5.5, 1))
            # ax.set_yticks(np.arange(.5, 5.5, 1))

            # Add grid lines (optional)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Re-add the circle to ensure it stays on the plot
            ax.add_patch(circle)
            plt.tight_layout()

            # Pause for animation effect
            plt.pause(4 / len(valences))
            plt.cla()
        pass
    continue