import numpy as np

grid = 5
nx = grid * 100 + 1
nz = grid * 100 + 1
dx = 40.0 / grid
dz = 40.0 / grid
n = nx * nz

model = {
    'x': np.zeros(n),
    'z': np.zeros(n),
    'rho': np.ones(n) * 2700.0,
    'vp': np.ones(n) * 3000.0,
    'vs': np.ones(n) * 1732.051
}

for i in range(nx):
    for j in range(nz):
        model['x'][i * nz + j] = i * dx
        model['z'][i * nz + j] = j * dz

npt = np.array([n], dtype='int32')

for m in model:
    with open(f'./model/proc000000_{m}.bin', 'w') as f:
        f.seek(0)
        npt.tofile(f)
        f.seek(4)
        model[m].astype('float32').tofile(f)
