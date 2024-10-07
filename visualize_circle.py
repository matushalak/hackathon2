import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import threading
import time

# Function to update valence and arousal externally
def set_valence_arousal(new_valence, new_arousal):
    global valence, arousal, positions
    valence = new_valence
    arousal = new_arousal
    print(f"Valence and arousal updated to Valence: {valence}, Arousal: {arousal}")
    # & associated random data
    valences = np.random.normal(valence, 0.25, 50)
    arousals = np.random.normal(arousal, 0.25, 50)
    positions =  np.column_stack((valences, arousals))

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
    global valence, arousal, positions, current_frame

    # Update current frame
    current_frame = frame % frames_per_cycle

    # Clear only the plotted data
    plt.cla()       # Clears line plots

    # Set the radius and create the circle
    radius = 3
    circle = Circle((3, 3), radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Get positions up to current frame
    idx = current_frame % 50
    current_positions = positions[:idx+1]
    v, a = positions[idx]

    # Calculate distances for dynamic sizing
    distances = np.sqrt((current_positions[:, 0] - 3)**2 + (current_positions[:, 1] - 3)**2)
    sizes = (1.5 - distances) * 200
    sizes = np.clip(sizes, 50, 200)

    # Determine colors based on quadrants
    colors = []
    for i in range(len(current_positions)):
        cmap = get_cmap(current_positions[i, 0], current_positions[i, 1])
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
    title_valence = valence
    title_arousal = arousal
    ax.set_title(f'Valence: {title_valence:.2f}, Arousal: {title_arousal:.2f}', fontsize=16)

    # Ensure spines remain at (3,3)
    ax.spines['left'].set_position(('data', 3))
    ax.spines['bottom'].set_position(('data', 3))

    # Set labels and aspect ratio
    ax.set_xlabel('Valence', fontsize=14, loc = 'left')
    ax.set_ylabel('Arousal', fontsize=14, loc = 'bottom')
    ax.set_aspect('equal')

    # Set limits
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

    plt.tight_layout()

def start_plot(v, a):
    # Set up the figure and axis
    global fig, ax
    fig, ax = plt.subplots(figsize=(10, 10))
    
    set_valence_arousal(v, a)
    
    # Define colormaps for each quadrant
    global frames_per_cycle, quadrant_cmaps
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


    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=interval_ms, repeat=True)
    # Use plt.pause() to make it non-blocking
    plt.draw()
    plt.pause(0.001)

def plot_live(v, a):
    start_plot(v, a)
    start = time.time()
    while time.time() - start < 4:
        plt.pause(0.001)  # Small pause to update the plot
    plt.close()

for i, j in zip([1, 5, 5, 1], [5, 1, 3, 3]):
    plot_live(i, j)