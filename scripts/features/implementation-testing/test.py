import numpy as np

from freq import fft


x = np.linspace(0, 4 * np.pi, 256)
y = np.sin(2 * np.pi * x) + 0.5 * np.cos(2 * np.pi * x/4) + 0.75 * np.random.rand(256)
nfft = 2 ** int(np.log2(x.size))

fF = fft(y, nfft)