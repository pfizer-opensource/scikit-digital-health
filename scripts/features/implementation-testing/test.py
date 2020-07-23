import numpy as np

np.random.seed(5)
x = np.around(np.random.rand(100, 1, 1), 2)


from sampent import sampleentropy, sampen2

sampleentropy(x, 3, 0.3)
sampen2(x[:, 0, 0], 3, 0.3)