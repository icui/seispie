from numba import cuda


@cuda.jit(device=True)
def idx():
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    k = tx + ty * bw
    return k


@cuda.jit(device=True)
def idxij(nz):
    k = idx()
    j = k % nz
    i = int((k - j) / nz)
    return k, i, j


@cuda.jit
def div_sy(dsy, sxy, szy, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsy.size:
        if i >= 2 and i < nx - 2:
            dsy[k] = 9 * (sxy[k] - sxy[k-nz]) / (8 * dx) - (sxy[k+nz] - sxy[k-2*nz]) / (24 * dx)
        else:
            dsy[k] = 0

        if j >= 2 and j < nz - 2:
            dsy[k] += 9 * (szy[k] - szy[k-1]) / (8 * dz) - (szy[k+1] - szy[k-2]) / (24 * dz)


@cuda.jit
def div_sy_c(dsx, dsz, dsy_c, syy_c, jk, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        if i >= 2 and i < nx - 2:
            dsz[k] -= 9 * (syy_c[k] - syy_c[k-nz]) / (8 * dx) - (syy_c[k+nz] - syy_c[k-2*nz]) / (24 * dx)

        if j >= 2 and j < nz - 2:
            dsx[k] += 9 * (syy_c[k] - syy_c[k-1]) / (8 * dz) - (syy_c[k+1] - syy_c[k-2]) / (24 * dz)

        dsy_c[k] -= 2 * syy_c[k] / jk[k]


@cuda.jit
def div_sxz(dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        if i >= 2 and i < nx - 2:
            dsx[k] = 9 * (sxx[k] - sxx[k-nz]) / (8 * dx) - (sxx[k+nz] - sxx[k-2*nz]) / (24 * dx)
            dsz[k] = 9 * (sxz[k] - sxz[k-nz]) / (8 * dx) - (sxz[k+nz] - sxz[k-2*nz]) / (24 * dx)
        else:
            dsx[k] = 0
            dsz[k] = 0

        if j >= 2 and j < nz - 2:
            dsx[k] += 9 * (sxz[k] - sxz[k-1]) / (8 * dz) - (sxz[k+1] - sxz[k-2]) / (24 * dz)
            dsz[k] += 9 * (szz[k] - szz[k-1]) / (8 * dz) - (szz[k+1] - szz[k-2]) / (24 * dz)


@cuda.jit
def stf_dsy(dsy, stf_y, src_id, isrc, it, nt):
    ib = cuda.blockIdx.x
    if isrc < 0 or isrc == ib:
        ks = ib * nt + it
        km = src_id[ib]
        dsy[km] += stf_y[ks]

@cuda.jit
def stf_dsxz(dsx, dsz, stf_x, stf_z, src_id, isrc, it, nt):
    ib = cuda.blockIdx.x
    if isrc < 0 or isrc == ib:
        ks = ib * nt + it
        km = src_id[ib]
        dsx[km] += stf_x[ks]
        dsz[km] += stf_z[ks]

@cuda.jit
def add_vy(vy, uy, dsy, rho, bound, dt, npt):
    k = idx()
    if k < npt:
        vy[k] = bound[k] * (vy[k] + dt * dsy[k] / rho[k])
        uy[k] += vy[k] * dt

@cuda.jit
def add_vxz(vx, vz, ux, uz, dsx, dsz, rho, bound, dt):
    k = idx()
    if k < vx.size:
        vx[k] = bound[k] * (vx[k] + dt * dsx[k] / rho[k])
        vz[k] = bound[k] * (vz[k] + dt * dsz[k] / rho[k])
        ux[k] += vx[k] * dt
        uz[k] += vz[k] * dt


@cuda.jit
def div_vy(dvydx, dvydz, vy, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < nx * nz:
        if i >= 1 and i < nx - 2:
            dvydx[k] = 9 * (vy[k+nz] - vy[k]) / (8 * dx) - (vy[k+2*nz] - vy[k-nz]) / (24 * dx)
        else:
            dvydx[k] = 0
        if j >= 1 and j < nz - 2:
            dvydz[k] = 9 * (vy[k+1] - vy[k]) / (8 * dz) - (vy[k+2] - vy[k-1]) / (24 * dz)
        else:
            dvydz[k] = 0

@cuda.jit
def div_vxz(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dvxdx.size:
        if i >= 1 and i < nx - 2:
            dvxdx[k] = 9 * (vx[k+nz] - vx[k]) / (8 * dx) - (vx[k+2*nz] - vx[k-nz]) / (24 * dx)
            dvzdx[k] = 9 * (vz[k+nz] - vz[k]) / (8 * dx) - (vz[k+2*nz] - vz[k-nz]) / (24 * dx)
        else:
            dvxdx[k] = 0
            dvzdx[k] = 0
        if j >= 1 and j < nz - 2:
            dvxdz[k] = 9 * (vx[k+1] - vx[k]) / (8 * dz) - (vx[k+2] - vx[k-1]) / (24 * dz)
            dvzdz[k] = 9 * (vz[k+1] - vz[k]) / (8 * dz) - (vz[k+2] - vz[k-1]) / (24 * dz)
        else:
            dvxdz[k] = 0
            dvzdz[k] = 0

@cuda.jit
def add_sy(sxy, szy, dvydx, dvydz, mu, dt, npt):
    k = idx()
    if k < npt:
        sxy[k] += dt * mu[k] * dvydx[k]
        szy[k] += dt * mu[k] * dvydz[k]

@cuda.jit
def add_sy_c(syx_c, syy_c, syz_c, vy_c, dvydx_c, dvydz_c, dvxdz, dvzdx, dvzdz, nu, jk, mu_c, nu_c, dt):
    k = idx()
    if k < syx_c.size:
        syy_c[k] += 2 * dt * nu[k] * (vy_c[k] - 0.5 * (dvzdx[k] - dvxdz[k]))
        syx_c[k] += dt / jk[k] * (mu_c[k] + nu_c[k]) * dvydx_c[k]
        syz_c[k] += dt / jk[k] * (mu_c[k] + nu_c[k]) * dvydz_c[k]

@cuda.jit
def add_sxz(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lam, mu, dt):
    k = idx()
    if k < sxx.size:
        sxx[k] += dt * ((lam[k] + 2 * mu[k]) * dvxdx[k] + lam[k] * dvzdz[k])
        szz[k] += dt * ((lam[k] + 2 * mu[k]) * dvzdz[k] + lam[k] * dvxdx[k])
        sxz[k] += dt * (mu[k] * (dvxdz[k] + dvzdx[k]))

@cuda.jit
def save_obs(obs, u, rec_id, it, nt, nx, nz):
    ib = cuda.blockIdx.x
    kr = ib * nt + it
    km = rec_id[ib]
    obs[kr] = u[km]


@cuda.jit
def interaction_muy(k_mu, dvydx, dvydx_fw, dvydz, dvydz_fw, ndt, nx, nz):
    k = idx()
    if k < nx * nz:
        k_mu[k] += (dvydx[k] * dvydx_fw[k] + dvydz[k] * dvydz_fw[k]) * ndt
