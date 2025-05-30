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


@cuda.jit(device=True)
def diff_x(v, i, k, dx, nx, nz):
    if i >= 2 and i < nx - 2:
        return 9 * (v[k] - v[k-nz]) / (8 * dx) - (v[k+nz] - v[k-2*nz]) / (24 * dx)
    else:
        return 0


@cuda.jit(device=True)
def diff_x1(v, i, k, dx, nx, nz):
    if i >= 1 and i < nx - 2:
        return 9 * (v[k+nz] - v[k]) / (8 * dx) - (v[k+2*nz] - v[k-nz]) / (24 * dx)
    else:
        return 0


@cuda.jit(device=True)
def diff_z(v, j, k, dz, nx, nz):
    if j >= 2 and j < nz - 2:
        return 9 * (v[k] - v[k-1]) / (8 * dz) - (v[k+1] - v[k-2]) / (24 * dz)
    else:
        return 0


@cuda.jit(device=True)
def diff_z1(v, j, k, dz, nx, nz):
    if j >= 1 and j < nz - 2:
        return 9 * (v[k+1] - v[k]) / (8 * dz) - (v[k+2] - v[k-1]) / (24 * dz)
    else:
        return 0


@cuda.jit
def div_sy(dsy, sxy, szy, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsy.size:
        dsy[k] = diff_x(sxy, i, k, dx, nx, nz) + diff_z(szy, j, k, dz, nx, nz)


@cuda.jit
def div_sxz(dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        dsx[k] = diff_x(sxx, i, k, dx, nx, nz) + diff_z(sxz, j, k, dz, nx, nz)
        dsz[k] = diff_x(sxz, i, k, dx, nx, nz) + diff_z(szz, j, k, dz, nx, nz)


@cuda.jit
def div_sxyz_c(dsx, dsz, dsy_c, syy_c, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        dsx[k] += diff_z(syy_c, j, k, dz, nx, nz)
        dsz[k] -= diff_x(syy_c, i, k, dx, nx, nz)
        dsy_c[k] -= 2 * syy_c[k]


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
        dvydx[k] = diff_x1(vy, i, k, dx, nx, nz)
        dvydz[k] = diff_z1(vy, j, k, dz, nx, nz)

@cuda.jit
def div_vxz(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dvxdx.size:
        dvxdx[k] = diff_x1(vx, i, k, dx, nx, nz)
        dvxdz[k] = diff_z1(vx, j, k, dz, nx, nz)
        dvzdx[k] = diff_x1(vz, i, k, dx, nx, nz)
        dvzdz[k] = diff_z1(vz, j, k, dz, nx, nz)

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
