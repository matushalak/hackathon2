import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class LiveValenceArousalPlot:
    def __init__(self):
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, 6)
        self.ax.set_ylim(0, 6)
        self.ax.set_aspect('equal')
        self.ax.spines['left'].set_position(('data', 3))
        self.ax.spines['bottom'].set_position(('data', 3))
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.set_xlabel('Valence', fontsize=14)
        self.ax.set_ylabel('Arousal', fontsize=14)

        # Create scatter plot elements with initial empty data
        self.scatter = self.ax.scatter([], [], s=[], color=[], edgecolors='none')
        self.highlight = self.ax.scatter([], [], s=[], color=[], edgecolors='black', linewidths=1)

        # Create circle boundary
        radius = 3
        circle = Circle((3, 3), radius, fill=False, color='black', linewidth=2)
        self.ax.add_patch(circle)

        # Initialize parameters
        self.valence = 3
        self.arousal = 3
        self.positions = np.zeros((50, 2))  # Initialize positions array
        self.update_positions()  # Set initial random positions
        self.quadrant_cmaps = {
            'Q1': plt.cm.Greens,
            'Q2': plt.cm.Reds,
            'Q3': plt.cm.Blues,
            'Q4': plt.cm.Purples
        }

        # Start animation
        self.ani = FuncAnimation(self.fig, self.animate, frames=200, interval=50, repeat=True)

        # Connect keyboard event to interactively update the plot
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def set_valence_arousal(self, new_valence, new_arousal):
        # Update valence and arousal and then recompute positions
        self.valence = new_valence
        self.arousal = new_arousal
        self.update_positions()
        print(f"Valence and arousal updated to Valence: {self.valence}, Arousal: {self.arousal}")

    def update_positions(self):
        """Update positions with Gaussian noise around the current valence and arousal."""
        self.positions = np.column_stack((
            np.random.normal(self.valence, 0.25, 50),
            np.random.normal(self.arousal, 0.25, 50)
        ))

    def get_cmap(self, v, a):
        if v > 3 and a > 3:
            return self.quadrant_cmaps['Q1']
        elif v < 3 and a > 3:
            return self.quadrant_cmaps['Q2']
        elif v < 3 and a < 3:
            return self.quadrant_cmaps['Q3']
        elif v > 3 and a < 3:
            return self.quadrant_cmaps['Q4']
        else:
            return plt.cm.Greys  # For points on the axis

    def animate(self, frame):
        idx = frame % 50
        current_positions = self.positions[:idx + 1]
        v, a = self.positions[idx]

        # Calculate sizes and colors
        distances = np.sqrt((current_positions[:, 0] - 3) ** 2 + (current_positions[:, 1] - 3) ** 2)
        sizes = (1.5 - distances) * 200
        sizes = np.clip(sizes, 50, 200)
        colors = [self.get_cmap(current_positions[i, 0], current_positions[i, 1])(0.6) for i in range(len(current_positions))]

        # Update scatter plot
        self.scatter.set_offsets(current_positions)
        self.scatter.set_sizes(sizes)
        self.scatter.set_color(colors)

        # Update highlight point
        self.highlight.set_offsets([[v, a]])
        self.highlight.set_sizes([sizes[-1]])
        self.highlight.set_color(self.get_cmap(v, a)(0.9))

        # Update title
        self.ax.set_title(f'Valence: {self.valence:.2f}, Arousal: {self.arousal:.2f}', fontsize=16)

    def on_key_press(self, event):
        """Handle key press events to modify the animation."""
        if event.key == 'up':
            self.set_valence_arousal(self.valence, self.arousal + 0.5)
        elif event.key == 'down':
            self.set_valence_arousal(self.valence, self.arousal - 0.5)
        elif event.key == 'right':
            self.set_valence_arousal(self.valence + 0.5, self.arousal)
        elif event.key == 'left':
            self.set_valence_arousal(self.valence - 0.5, self.arousal)
        print(f"Key pressed: {event.key}")

    def start(self):
        plt.show()