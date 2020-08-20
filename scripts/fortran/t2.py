if __name__ == '__main__':
    from pfi import real_fft
    from fft import _pocketfft_internal as cpfi
    import numpy as np
    from sys import argv
    
    np.random.seed(5)
    
    if len(argv) == 2:
        n = int(argv[1])
        if (n & (n-1)) == 0:
            x = np.random.rand(n)
        else:
            raise ValueError('not power of 2')
    else:
        x = np.random.rand(64)
    
    fres, ier = real_fft.execute_real_forward(x, 1.0)
    if ier!=0:
        raise ValueError('Error in fortran extension')
    res = cpfi.execute(x, True, True, 1.0)
#     print('\n', res[:3])
#     print(fres.view(np.complex128)[:3])
    
    print(np.allclose(fres.view(np.complex128), res))
