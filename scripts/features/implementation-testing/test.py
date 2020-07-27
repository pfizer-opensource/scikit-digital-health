import numpy as np

from perment import permutationentropy


np.random.seed(17)
n = 20
x = np.around(np.random.rand(n, 1, 1), 2)

permutationentropy(x, 3, 1, False)