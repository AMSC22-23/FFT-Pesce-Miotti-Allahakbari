
'''
This short script is used to better understand the effect of multiple types
of padding, before we choose how to implement them in cpp, namely:

+ Regular zero-padding.
+ Padding with a periodic continuation of the signal.

This script will also experiment with restricting the output signal to be
the same size as the input signal, by trivially removing elements in an
uniform pattern.
'''

import numpy as np
import matplotlib.pyplot as plt

def zero_pad(signal, new_width):
    new_signal = signal.copy()
    for _ in range(new_width - len(signal)):
        new_signal.append(0)

    return new_signal

def periodic_pad(signal, new_width):
    new_signal = signal.copy()
    for i in range(new_width - len(signal)):
        new_signal.append(signal[i % len(signal)])

    return new_signal

def restrict(signal, new_width, height_multiplier = 1):
    restricted = []
    for i in range(new_width):
        new_index = int(i * len(signal) / new_width)
        restricted.append(signal[new_index] * height_multiplier)

    return restricted

def plot(signal, title):
    plt.figure()
    plt.plot(signal)
    plt.title(title)    

# Edit these parameters to experiment with different signals.
original_width = 100
new_width = 200

# We begin by creating a signal by summing some random sine waves.
signal = []
for i in range(original_width):
    signal.append(np.sin(i * 0.1) + np.sin(i * 0.2) + np.sin(i * 0.3))

# Uncomment this to experiment with a random signal instead.
""" # We begin by creating a signal by generating random numbers.
signal = []
for i in range(original_width):
    signal.append(np.random.rand()) """

# We then pad the signal with zeros.
zero_padded = zero_pad(signal, new_width)

# We then pad the signal with a periodic continuation of the signal.
periodic_padded = periodic_pad(signal, new_width)

# We can now apply numpy's fft to the signals.
regular_fft = np.fft.fft(signal)
zero_padded_fft = np.fft.fft(zero_padded)
periodic_padded_fft = np.fft.fft(periodic_padded)

# Let's restrict the periodic padded signal to the same size as the input.
ratio = original_width / new_width
restricted = restrict(periodic_padded_fft, original_width, ratio)

# We'll now plot the input signal, the regular non-padded fft,
# the zero padded result, and the periodic result.
plot(signal, "Input signal")
plot(regular_fft, "Regular FFT")
plot(zero_padded_fft, "Zero padded FFT (should be incorrect)")
plot(restricted, "Periodic padded & restricted FFT")

plt.show()