from pfi import real_fft
# from test import test
import numpy as np
import os

os.remove('tw_f.txt')
real_fft.execute_real_forward(np.random.rand(64), 5.0)

# test()
