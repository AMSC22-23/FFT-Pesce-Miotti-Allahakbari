import sys
import numpy as np
import matplotlib.pyplot as plt

# Check for correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python3 visualize-fft.py <initial-sequence> <result>")
    exit(1)

# Read data from file
sequence = np.genfromtxt(sys.argv[1], delimiter=',')
result = np.genfromtxt(sys.argv[2], delimiter=',')

# Turn complex numbers into magnitudes
sequence = [np.sqrt(x[0]**2 + x[1]**2) for x in sequence]
result = [np.sqrt(x[0]**2 + x[1]**2) for x in result]

# Set up plot for initial sequence
plt.subplot(2, 1, 1)
plt.plot(sequence)
plt.title("Initial Sequence")
plt.xlabel("Index")
plt.ylabel("Magnitude")

# Set up plot for result
plt.subplot(2, 1, 2)
plt.plot(result)
plt.title("Result")
plt.xlabel("Index")
plt.ylabel("Magnitude")

# Display plot
plt.tight_layout()
plt.show()