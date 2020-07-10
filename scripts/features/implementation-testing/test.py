from scipy.misc import electrocardiogram

x1d = electrocardiogram()
x3d = x1d.reshape((1, -1, 1))


from sigent import fsignalentropy

for i in range(3):
    res = fsignalentropy(x3d)
    print(res)