from acorr import autocorr as fcac
import numpy as np

x = np.random.rand(50000, 150, 3)
xc = np.ascontiguousarray(x.transpose([0, 2, 1]))
xf = np.asfortranarray(x.transpose([1, 2, 0]))

fcac(xc, 1, 1)