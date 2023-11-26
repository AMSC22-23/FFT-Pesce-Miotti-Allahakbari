import sys
import numpy as np
import matplotlib.pyplot as plt

# Check that at least one argument is provided
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <sequence 1> <sequence 2> ...")
    exit(1)

# Load all csv files, (real, imaginary) pairs  
sequences = []
for i in range(1, len(sys.argv)):
    sequences.append(np.genfromtxt(sys.argv[i], delimiter=','))

# Plot each sequence
for i in range(len(sequences)):
    plt.subplot(len(sequences), 1, i + 1)
    plt.plot([x[0] for x in sequences[i]])
    plt.plot([x[1] for x in sequences[i]])
    plt.title("Sequence " + str(i + 1) + ": " + sys.argv[i + 1])
    plt.xlabel("Index")
    plt.ylabel("Magnitude")
    plt.legend(["Real", "Imaginary"])

plt.tight_layout()
plt.show()