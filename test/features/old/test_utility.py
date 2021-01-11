# import numpy as np
#
# from skimu.features.lib.extensions._utility import cf_rfft
#
#
# def fft1(x):
#     """
#     Truth value for FFT. Taken from numpy
#
#     numpy/fft/tests/test_pocketfft.py
#     """
#     L = len(x)
#     phase = -2j*np.pi*(np.arange(L)/float(L))
#     phase = np.arange(L).reshape(-1, 1) * phase
#     return np.sum(x*np.exp(phase), axis=1)
#
#
# class TestRFFT:
#     def test(self):
#         """
#         Taken from numpy. Combination of fft and rfft tests to have a truth value
#         for rfft that is not fft computation
#         """
#         for n in [32, 64, 128, 1024]:
#             for i in range(25):  # repeat each test a few times
#                 x = np.random.random(n)
#
#                 assert np.allclose(
#                     fft1(x)[:(n//2 + 1)],
#                     cf_rfft(x, n//2).view(np.complex128),
#                     atol=1e-6
#                 )
#                 assert np.allclose(
#                     np.fft.rfft(x, n),
#                     cf_rfft(x, n//2).view(np.complex128),
#                     atol=1e-6
#                 )
