import sys
import csv

import matplotlib.pyplot as plt

import numpy as np

# Check that the user has provided the correct number of arguments
if len(sys.argv) != 2:
    print("Usage: python visualize-performance.py times.csv")
    sys.exit(1)

# Read the CSV file
with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rows = list(reader)

# Set thread number on x-axis
x_axis = rows[0][1:]

# Create one figure for each category
for i in range(1, len(rows)):
    # Create a new figure
    plt.figure(i)

    # Turn the values into floats
    times = [float(x) for x in rows[i][1:]]
    plt.plot(times)

    # Add title and axis labels
    plt.title(rows[i][0])
    plt.xlabel('Number of threads')
    plt.xticks(np.arange(len(x_axis)), x_axis)

# Show the plots
plt.show()