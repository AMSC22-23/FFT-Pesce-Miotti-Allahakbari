import sys
import numpy as np
import matplotlib.pyplot as plt

# Check for correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python3 visualize-fft.py <initial-sequence> <result>")
    exit(1)

# Read data from file
data = np.genfromtxt(sys.argv[1], delimiter=',')
data2 = np.genfromtxt(sys.argv[2], delimiter=',')

# TODO: Plot data