import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# Create a new figure
fig, ax = plt.subplots()

# Draw a simple line plot
ax.plot([0, 1, 2, 3], [0, 1, 4, 9])

# Set title and labels
ax.set_title("Test Plot")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")

# Display the plot.
plt.show()

x = 1
