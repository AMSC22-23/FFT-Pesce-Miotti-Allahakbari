
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
periodic_padded_fft = np.fft.fft(periodic_padded)

# Let's restrict the zero padded signal to the same size as the input.
ratio = original_width / new_width
zero_padded_fft = np.fft.fft(zero_padded)

# No need to change the height of the signal, since we're not repeating it.
zero_padded_fft = restrict(zero_padded_fft, original_width)

# Let's restrict the periodic padded signal to the same size as the input.
periodic_padded_fft = restrict(periodic_padded_fft, original_width, ratio)

# We'll now plot the input signal, the regular non-padded fft,
# the zero padded result, and the periodic result.
plot(signal, "Input signal")
plot(regular_fft, "Regular FFT")
plot(zero_padded_fft, "Zero padded & restricted FFT")
plot(periodic_padded_fft, "Periodic padded & restricted FFT")

plt.show()

'''
Acknowledgements:

The periodic padding idea does not work as well as I had hoped. The reason
is that, if the input is completely random, then the periodic padding combined
with the less-than-ideal restriction will not coincide with the regular fft.

According to https://www.bitweenie.com/listings/fft-zero-padding/ we should consider
that the non-restricted periodic padding is nevertheless a much more precise
version of the standard fft, with a higher resolution. This is because the
periodic padding effectively increases the amount of samples in the signal,
while keeping the same frequency shape.

My conclusion is that the restriction part is the most problematic, especially if the
ratio between the original width and the new width is not an integer. This is because
the restriction will then have to interpolate badly between the values, which will
introduce errors. Try to experiment with the ratio parameter in the restrict function
(try 100/250) and note the visible errors in the plot, even with the sinusoidal input.

IMPORTANT ACKNOWLEDGEMENT:
Although the periodic padding method had shown some promise, after experimenting with 
ratio = 129/256, I realized that this whole method might be completely useless. The 
reason is that on the source website above, we assume that we know our full signal to 
begin with, which is not the case in our application. Therefore, we cannot simply pad 
the signal periodically, because we don't know what the signal looks like after the 
last sample.

Additionally, this answer:
https://math.stackexchange.com/questions/256791/zero-padding-data-for-fft
seems to suggest that true fast padding is never accurate. The answer even
goes as far as to suggest that we should do the IFFT, then DFT, which is definitely
not what we want to do.
'''