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

# Set up plot for initial sequence
plt.subplot(2, 1, 1)
plt.plot([x[0] for x in sequence])
plt.plot([x[1] for x in sequence])
plt.title("Initial Sequence")
plt.xlabel("Index")
plt.ylabel("Magnitude")
plt.legend(["Real", "Imaginary"])

# Set up plot for result
plt.subplot(2, 1, 2)
plt.plot([x[0] for x in result])
plt.plot([x[1] for x in result])
plt.title("Result")
plt.xlabel("Index")
plt.ylabel("Magnitude")
plt.legend(["Real", "Imaginary"])

# Display plot
plt.tight_layout()
plt.show()