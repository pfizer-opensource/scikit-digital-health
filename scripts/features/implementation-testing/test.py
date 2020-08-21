from sparc2 import sparc_1d as fsparc2_1d
import numpy as np

x = np.random.rand(150)
fs = 50.0
cut = 10.0
pad = 4
thresh = 0.05
nfft = int(pow(2, np.ceil(np.log2(150)) + pad))

a = fsparc2_1d(x, fs, nfft, cut, thresh)
b = fsparc2_1d(x, fs, nfft, cut, thresh)
print(a, b)