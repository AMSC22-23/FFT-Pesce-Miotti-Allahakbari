import sys
import numpy as np

# Check for correct amount of arguments
if len(sys.argv) != 3:
    print(f"Usage: python3 {sys.argv[0]} <initial sequence> <result sequence>")
    exit(1)

# Load sequences
initial = np.genfromtxt(sys.argv[1], delimiter=',')
result = np.genfromtxt(sys.argv[2], delimiter=',')

# Convert to complex numbers
initial = [complex(x[0], x[1]) for x in initial]
result = [complex(x[0], x[1]) for x in result]

# Check that the sequences are the same length
if len(initial) != len(result):
    print("Sequences are not the same length.")
    exit(1)

# Compute the fft of the initial sequence
fft = np.fft.fft(initial)

# Get the error between the fft and the result
error = np.abs(fft - result)

# Print the max error
print(f"Max error: {np.max(error):.2e}")