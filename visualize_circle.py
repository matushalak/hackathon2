import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Set the radius and create the circle
radius = 3
circle = Circle((3, 3), radius, fill=False, color='black', linewidth=2)
ax.add_patch(circle)

# Set limits and aspect ratio
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)
ax.set_aspect('equal')

# Adjust the spines to cross at (3,3)
ax.spines['left'].set_position(('data', 3))
ax.spines['bottom'].set_position(('data', 3))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Set labels
ax.set_xlabel('Valence', fontsize=14)
ax.set_ylabel('Arousal', fontsize=14)

# Define colormaps for each quadrant
quadrant_cmaps = {
    'Q1': plt.cm.Greens,
    'Q2': plt.cm.Reds,
    'Q3': plt.cm.Blues,
    'Q4': plt.cm.Purples
}

# Initialize variables
num_points = 200
interval_ms = 20  # Interval between frames in milliseconds
frames_per_cycle = int(4000 / interval_ms)  # Number of frames per 4-second cycle
start_time = time.time()

# Generate initial random data
valence, arousal = np.random.randint(1, 5, 2)
valences = np.random.normal(valence, 0.25, num_points)
arousals = np.random.normal(arousal, 0.25, num_points)
positions = np.column_stack((valences, arousals))
current_frame = 0

# Function to determine color based on quadrant
def get_cmap(v, a):
    if v > 3 and a > 3:
        return quadrant_cmaps['Q1']
    elif v < 3 and a > 3:
        return quadrant_cmaps['Q2']
    elif v < 3 and a < 3:
        return quadrant_cmaps['Q3']
    elif v > 3 and a < 3:
        return quadrant_cmaps['Q4']
    else:
        return plt.cm.Greys  # For points on the axis

# Animation function
def animate(frame):
    global valence, arousal, valences, arousals, positions, start_time, current_frame
    
    # Update current frame
    current_frame = frame % frames_per_cycle

    # Check if 4 seconds have passed to update initial coordinates
    if current_frame == 0:
        # Generate new initial random data
        valence, arousal = np.random.randint(1, 5, 2)
        valences = np.random.normal(valence, 0.25, num_points)
        arousals = np.random.normal(arousal, 0.25, num_points)
        positions = np.column_stack((valences, arousals))
    
    # Clear the axis but keep the circle and spines
    plt.cla()
    ax.add_patch(circle)

    ax.set_xlabel('Valence', fontsize=14, loc = 'left')
    ax.set_ylabel('Arousal', fontsize=14, loc = 'bottom')

    # Get positions up to current frame
    idx = current_frame % num_points
    current_positions = positions[:idx+1]
    v, a = positions[idx]

    # Calculate distances for dynamic sizing
    distances = np.sqrt((current_positions[:, 0] - 3)**2 + (current_positions[:, 1] - 3)**2)
    sizes = (1.5 - distances) * 200
    sizes = np.clip(sizes, 50, 200)

    # Determine colors based on quadrants
    colors = []
    for idx in range(len(current_positions)):
        cmap = get_cmap(current_positions[idx, 0], current_positions[idx, 1])
        color = cmap(0.6)  # Use a consistent shade
        colors.append(color)

    # Fading effect
    alphas = np.linspace(0.2, 1, len(current_positions))
    colors = np.array(colors)
    colors[:, -1] = alphas  # Set alpha channel

    # Plot the points
    ax.scatter(current_positions[:, 0], current_positions[:, 1], s=sizes, color=colors, edgecolors='none')

    # Highlight the current point
    cmap = get_cmap(v, a)
    ax.scatter(v, a, s=sizes[-1], color=cmap(0.9), edgecolors='black', linewidths=1)

    # Set the title with current valence and arousal
    ax.set_title(f'Valence: {valence}, Arousal: {arousal}', fontsize=16)

    # Ensure spines remain at (3,3)
    ax.spines['left'].set_position(('data', 3))
    ax.spines['bottom'].set_position(('data', 3))

    plt.tight_layout()
    # Set grid
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=100000, interval=interval_ms, repeat=True)

# Show the plot
plt.show()