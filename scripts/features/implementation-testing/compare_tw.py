import numpy as np

with open('tw_f.txt', 'r') as f:
    flines = f.readlines()
with open('tw_c.txt', 'r') as f:
    clines = f.readlines()
    
cvals = np.zeros((54, 2))

for i, line in enumerate(clines):
    if line != '':
        cvals[:, i] = [np.float(i) for i in line.split(',')[:-1]]

fvals = np.zeros((54, 2))
for i, line in enumerate(flines[:2]):  # last lines are not necessary
    fvals[:, i] = [np.float(i) for i in line.split(' ') if i != '']

print(cvals - fvals)
