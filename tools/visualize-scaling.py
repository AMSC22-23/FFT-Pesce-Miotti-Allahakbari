
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np

# Check that the user has provided the correct number of arguments
if len(sys.argv) != 2:
    print("Usage: python visualize-scaling.py scaling.csv")
    sys.exit(1)

# Read the CSV file
with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rows = list(reader)

# Flip rows and columns
cols = list(zip(*rows[1:]))

x_axis = cols[0]
functions = cols[1:]
labels = rows[0][1:]

# Set x-axis labels
plt.xticks(np.arange(len(x_axis)), x_axis)

# Plot the data, considering functions one by one
for function in functions:
    function = [float(x) for x in function]
    plt.plot(function)

# Add legend
plt.legend(labels)

# Add title and axis labels
plt.xlabel('Power of 2')
plt.ylabel('Time')

# Show the plot
plt.show()