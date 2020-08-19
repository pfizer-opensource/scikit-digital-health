if __name__ == '__main__':
    from pfi import real_fft
    import numpy as np
    from sys import argv
    
    if len(argv) == 2:
        real_fft.execute_real_forward(np.random.rand(int(argv[1])), 5.0)
    else:
        real_fft.execute_real_forward(np.random.rand(64), 5.0)
