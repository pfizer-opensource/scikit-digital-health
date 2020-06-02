from wfft import win_fft as wfft
import numpy as np

x = np.random.rand(1500)

for i in range(100):
    f = wfft(x, 512, 400, 300)
  