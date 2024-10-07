import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import threading
import time

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

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

# Initialize valence and arousal with default values
valence = 3.0
arousal = 3.0

# Lock for thread safety
valence_arousal_lock = threading.Lock()

# Function to update valence and arousal externally
def set_valence_arousal(new_valence, new_arousal):
    global valence, arousal
    with valence_arousal_lock:
        valence = new_valence
        arousal = new_arousal
    print(f"Valence and arousal updated to Valence: {valence}, Arousal: {arousal}")

# Generate initial random data
def generate_positions():
    with valence_arousal_lock:
        v = valence
        a = arousal
    valences = np.random.normal(v, 0.25, num_points)
    arousals = np.random.normal(a, 0.25, num_points)
    return np.column_stack((valences, arousals))

positions = generate_positions()
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
    global valence, arousal, positions, current_frame

    # Update current frame
    current_frame = frame % frames_per_cycle

    # Check if it's time to update data
    if current_frame == 0:
        # Automatically generate new random valence and arousal
        with valence_arousal_lock:
            valence = np.random.uniform(1, 5)
            arousal = np.random.uniform(1, 5)
            print(f"Automatically updated to Valence: {valence:.2f}, Arousal: {arousal:.2f}")
        # Generate new data using current valence and arousal
        positions = generate_positions()

    # Clear only the plotted data
    plt.cla()       # Clears line plots

    # Set the radius and create the circle
    radius = 3
    circle = Circle((3, 3), radius, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

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
    with valence_arousal_lock:
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

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=100000, interval=interval_ms, repeat=True)

# Example function to modify valence and arousal externally
def external_update():
    # Wait for some time and then update valence and arousal
    time.sleep(10)  # Wait for 10 seconds
    set_valence_arousal(4.5, 2.5)  # Update to new values
    # You can add more updates here if desired
    time.sleep(10)
    set_valence_arousal(2.0, 4.0)

# Start the external update thread
update_thread = threading.Thread(target=external_update, daemon=True)
update_thread.start()

# Show the plot
plt.show()