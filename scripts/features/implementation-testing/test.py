import numpy as np

xb = np.random.rand(2, 16, 1)
xc = np.ascontiguousarray(xb.transpose([0, 2, 1]))


from sigent import fsignalentropy, fsignalentropy2

fsignalentropy(xb.T)
fsignalentropy2(xc.T)