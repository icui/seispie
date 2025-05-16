import numpy as np
import math

from os import path, makedirs
from time import time
from numba import cuda

from seispie.solver.base import base
from seispie.solver.source.ricker import ricker
from seispie.solver.misfit.waveform import waveform

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
def vps2lm(lam, mu, rho):
	k = idx()
	if k < lam.size:
		vp = lam[k]
		vs = mu[k]

		if vp > vs:
			lam[k] = rho[k] * (vp * vp - 2 * vs * vs)
		else:
			lam[k] = 0

		mu[k] = rho[k] * vs * vs

@cuda.jit
def lm2vps(vp, vs, rho):
	k = idx()
	if k < vp.size:
		lam = vp[k]
		mu = vs[k]

		vp[k] = math.sqrt((lam + 2 * mu) / rho[k])
		vs[k] = math.sqrt(mu / rho[k])

@cuda.jit
def clear_field(field):
	k = idx()
	if k < field.size:
		field[k] = 0

@cuda.jit
def set_bound(bound, width, alpha, left, right, bottom, top, nx, nz):
	k, i, j = idxij(nz)
	if k < bound.size:
		bound[k] = 1

		if left and i + 1 < width:
			aw = alpha * (width - i - 1)
			bound[k] *= math.exp(-aw * aw)


		if right and i > nx - width:
			aw = alpha * (width + i - nx)
			bound[k] *= math.exp(-aw * aw)

		if bottom and j > nz - width:
			aw = alpha * (width + j - nz)
			bound[k] *= math.exp(-aw * aw)

		if top and j + 1 < width:
			aw = alpha * (width - j - 1)
			bound[k] *= math.exp(-aw * aw)

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
def div_sy_c(dsx, dsz, dsy_c, syy_c, jk, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < dsx.size:
		if i >= 2 and i < nx - 2:
			dsz[k] -= 9 * (syy_c[k] - syy_c[k-nz]) / (8 * dx) - (syy_c[k+nz] - syy_c[k-2*nz]) / (24 * dx)

		if j >= 2 and j < nz - 2:
			dsx[k] += 9 * (syy_c[k] - syy_c[k-1]) / (8 * dz) - (syy_c[k+1] - syy_c[k-2]) / (24 * dz)

		dsy_c[k] -= 2 * syy_c[k] / jk[k]

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

@cuda.jit(device=True)
def gaussian(x, sigma):
	return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-x * x / (2 * sigma * sigma))

@cuda.jit
def init_gausian(gsum, sigma, nx, nz):
	k, i, j = idxij(nz)
	if k < nx * nz:
		sumx = 0
		for n in range(nx):
			sumx += gaussian(i - n, sigma)

		sumz = 0
		for n in range(nz):
			sumz += gaussian(j - n, sigma)

		gsum[k] = sumx * sumz

@cuda.jit
def apply_gauxxian_x(data, gtmp, sigma, nx, nz):
	k, i, j = idxij(nz)
	if k < nx * nz:
		sumx = 0
		for n in range(nx):
			sumx += gaussian(i - n, sigma) * data[n * nz + j]

		gtmp[k] = sumx

@cuda.jit
def apply_gauxxian_z(data, gtmp, gsum, sigma, nx, nz):
	k, i, j = idxij(nz)
	if k < nx * nz:
		sumz = 0
		for n in range(nz):
			sumz += gaussian(j - n, sigma) * gtmp[i * nz + n]

		data[k] = sumz / gsum[k]

class fdm(base):
	def setup(self, workflow):
		# FIXME validate config
		self.stream = cuda.stream()
		self.nt = int(self.config['nt'])
		self.dt = float(self.config['dt'])
		self.sh = 1 if self.config['sh'] == 'yes' else 0
		self.psv = 1 if self.config['psv'] == 'yes' else 0
		self.spin = 1 if self.config['spin'] == 'yes' else 0
		self.sae = 0
		self.nsa = 0

	def import_sources(self):
		src = np.loadtxt(self.path['sources'], ndmin=2)
		nsrc = self.nsrc = src.shape[0]
		src_id = np.zeros(nsrc, dtype='int32')

		stf_x = np.zeros(nsrc * self.nt, dtype='float32')
		stf_y = np.zeros(nsrc * self.nt, dtype='float32')
		stf_z = np.zeros(nsrc * self.nt, dtype='float32')

		for isrc in range(nsrc):
			src_x = int(np.round(src[isrc][0] / self.dx))
			src_z = int(np.round(src[isrc][1] / self.dz))
			src_id[isrc] = src_x * self.nz + src_z

			for it in range(0, self.nt):
				istf = isrc * self.nt + it
				stf_x[istf], stf_y[istf], stf_z[istf] = ricker(it * self.dt, *src[isrc][3:])

		# allocate array
		stream = self.stream

		self.src_id = cuda.to_device(src_id, stream=stream)
		self.stf_x = cuda.to_device(stf_x, stream=stream)
		self.stf_y = cuda.to_device(stf_y, stream=stream)
		self.stf_z = cuda.to_device(stf_z, stream=stream)

	def import_stations(self):
		rec = np.loadtxt(self.path['stations'], ndmin=2)
		nrec = self.nrec = rec.shape[0]
		rec_id = np.zeros(nrec, dtype='int32')

		# allocate array
		stream = self.stream

		for irec in range(nrec):
			rec_x = int(np.round(rec[irec][0] / self.dx))
			rec_z = int(np.round(rec[irec][1] / self.dz))
			rec_id[irec] = rec_x * self.nz + rec_z

		obs = np.zeros(nrec * self.nt, dtype='float32')
		self.rec_id = cuda.to_device(rec_id, stream=stream)
		self.obs_x = cuda.to_device(obs, stream=stream)
		self.obs_y = cuda.to_device(obs, stream=stream)
		self.obs_z = cuda.to_device(obs, stream=stream)

	def import_model(self, model_true):
		""" import model
		"""
		model = dict()
		model_dir = self.path['model_true'] if model_true else self.path['model_init']
		model_params = ['x', 'z', 'lambda', 'mu', 'nu', 'j', 'lambda_c', 'mu_c', 'nu_c', 'rho'] if self.spin else ['x', 'z', 'vp', 'vs', 'rho']

		for name in model_params:
			filename = path.join(model_dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				if not hasattr(self, 'npt'):
					f.seek(0)
					self.npt = np.fromfile(f, dtype='int32', count=1)

				f.seek(4)
				model[name] = np.fromfile(f, dtype='float32')

		npt = self.npt[0]
		ntpb = int(self.config['threads_per_block'])
		nb = int(np.ceil(npt / ntpb))
		self.dim = nb, ntpb

		x = model['x']
		z = model['z']
		lx = x.max() - x.min()
		lz = z.max() - z.min()
		nx = self.nx = int(np.rint(np.sqrt(npt * lx / lz)))
		nz = self.nz = int(np.rint(np.sqrt(npt * lz / lx)))
		dx = self.dx = lx / (nx - 1)
		dz = self.dz = lz / (nz - 1)

		# allocate array
		stream = self.stream
		zeros = np.zeros(npt, dtype='float32')

		# change parameterization
		if self.spin:
			self.lam = cuda.to_device(model['lambda'], stream=stream)
			self.mu = cuda.to_device(model['mu'], stream=stream)
			self.nu = cuda.to_device(model['nu'], stream=stream)
			self.j = cuda.to_device(model['j'], stream=stream)
			self.lam_c = cuda.to_device(model['lambda_c'], stream=stream)
			self.mu_c = cuda.to_device(model['mu_c'], stream=stream)
			self.nu_c = cuda.to_device(model['nu_c'], stream=stream)
		else:
			self.lam = cuda.to_device(model['vp'], stream=stream)
			self.mu = cuda.to_device(model['vs'], stream=stream)
			vps2lm[self.dim](self.lam, self.mu, self.rho)

		self.rho = cuda.to_device(model['rho'], stream=stream)
		self.bound = cuda.to_device(zeros, stream=stream) # absorbing boundary

		abs_left = 1 if self.config['abs_left'] == 'yes' else 0
		abs_right = 1 if self.config['abs_right'] == 'yes' else 0
		abs_top = 1 if self.config['abs_top'] == 'yes' else 0
		abs_bottom = 1 if self.config['abs_bottom'] == 'yes' else 0

		set_bound[self.dim](
			self.bound, int(self.config['abs_width']), float(self.config['abs_alpha']),
			abs_left, abs_right, abs_bottom, abs_top, nx, nz
		)

		dats = []

		if self.sh:
			dats += ['vy', 'uy', 'sxy', 'szy', 'dsy', 'dvydx', 'dvydz']

		if self.psv:
			dats += [
				'vx', 'vz', 'ux', 'uz', 'sxx', 'szz', 'sxz',
				'dsx', 'dsz','dvxdx', 'dvxdz', 'dvzdx', 'dvzdz'
			]

			if self.spin:
				dats += ['vy_c', 'uy_c', 'syx_c', 'syy_c', 'syz_c', 'dsy_c', 'dvydx_c', 'dvydz_c']

		for dat in dats:
			setattr(self, dat, cuda.to_device(zeros, stream=stream))

		# FIXME interpolate model

		# write coordinate file
		if self.config['save_coordinates']:
			self.export_field(x, 'x')
			self.export_field(z, 'z')

	def export_field(self, field, name, it=0):
		name = 'proc%06d_%s' % (it, name)
		with open(self.path['output'] + '/' + name + '.bin', 'w') as f:
			f.seek(0)
			self.npt.tofile(f)

			f.seek(4)
			field.tofile(f)

	def setup_adjoint(self):
		self.sae = int(self.config['adjoint_interval'])
		self.nsa = int(self.nt / self.sae)
		stream = self.stream
		adstf = np.zeros(self.nt * self.nrec, dtype='float32')

		nsa = self.nsa
		npt = self.nx * self.nz
		zeros = np.zeros(npt, dtype='float32')

		if self.sh:
			self.adstf_y = cuda.to_device(adstf, stream=stream)
			self.dvydx_fw = cuda.to_device(zeros, stream=stream)
			self.dvydz_fw = cuda.to_device(zeros, stream=stream)

			self.uy_fwd = np.zeros([nsa, npt], dtype='float32')
			self.vy_fwd = np.zeros([nsa, npt], dtype='float32')

		if self.psv:
			self.adstf_x = cuda.to_device(adstf, stream=stream)
			self.adstf_z = cuda.to_device(adstf, stream=stream)
			self.dvxdx_fw = cuda.to_device(zeros, stream=stream)
			self.dvxdz_fw = cuda.to_device(zeros, stream=stream)
			self.dvzdx_fw = cuda.to_device(zeros, stream=stream)
			self.dvzdz_fw = cuda.to_device(zeros, stream=stream)

			self.ux_fwd = np.zeros([nsa, npt], dtype='float32')
			self.vx_fwd = np.zeros([nsa, npt], dtype='float32')
			self.uz_fwd = np.zeros([nsa, npt], dtype='float32')
			self.vz_fwd = np.zeros([nsa, npt], dtype='float32')

		self.k_lam = cuda.to_device(zeros, stream=stream)
		self.k_mu = cuda.to_device(zeros, stream=stream)
		self.k_rho = cuda.to_device(zeros, stream=stream)

		self.gsum = cuda.to_device(zeros, stream=stream)
		self.gtmp = cuda.to_device(zeros, stream=stream)
		self.sigma = int(self.config['smooth'])
		init_gausian[self.dim](self.gsum, self.sigma, self.nx, self.nz)

	def _compute_misfit(self, comp, h_syn):
		stream = self.stream
		syn = cuda.to_device(h_syn, stream)
		obs = getattr(self, 'obs_' + comp)
		adstf = getattr(self, 'adstf_' + comp)
		misfit = waveform(syn, obs, adstf, self.nt, self.dt, self.nrec, stream)
		return misfit

	def clear_kernels(self):
		dim = self.dim

		clear_field[dim](self.k_lam)
		clear_field[dim](self.k_mu)
		clear_field[dim](self.k_rho)

	def clear_wavefields(self):
		dim = self.dim

		if self.sh:
			clear_field[dim](self.vy)
			clear_field[dim](self.uy)
			clear_field[dim](self.sxy)
			clear_field[dim](self.szy)

		if self.psv:
			clear_field[dim](self.vx)
			clear_field[dim](self.vz)
			clear_field[dim](self.ux)
			clear_field[dim](self.uz)
			clear_field[dim](self.sxx)
			clear_field[dim](self.szz)
			clear_field[dim](self.sxz)

	def run_forward(self):
		stream = self.stream
		dim = self.dim
		sh = self.sh
		psv = self.psv
		spin = self.spin

		nx = self.nx
		nz = self.nz
		nt = self.nt
		dx = self.dx
		dz = self.dz
		dt = self.dt

		if self.config['combine_sources'] == 'yes':
			isrc = -1
		else:
			isrc = self.taskid

		npt = self.npt[0]
		out = np.zeros(npt, dtype='float32')
		sfe = int(self.config['save_snapshot'])
		sae = self.sae

		self.clear_wavefields()

		for it in range(self.nt):
			isa = -1
			if sae and (it + 1) % sae == 0:
				isa = int(self.nsa - (it + 1) / sae)

			if isa >= 0:
				if sh:
					self.uy.copy_to_host(self.uy_fwd[isa], stream=stream)

				if psv:
					self.ux.copy_to_host(self.ux_fwd[isa], stream=stream)
					self.uz.copy_to_host(self.uz_fwd[isa], stream=stream)

			if sh:
				div_sy[dim](self.dsy, self.sxy, self.szy, dx, dz, nx, nz)
				stf_dsy[self.nsrc, 1](self.dsy, self.stf_y, self.src_id, isrc, it, nt)
				add_vy[dim](self.vy, self.uy, self.dsy, self.rho, self.bound, dt, npt)
				div_vy[dim](self.dvydx, self.dvydz, self.vy, dx, dz, nx, nz)
				add_sy[dim](self.sxy, self.szy, self.dvydx, self.dvydz, self.mu, dt, npt)
				save_obs[self.nrec, 1](self.obs_y, self.uy, self.rec_id, it, nt, nx, nz)

			if psv:
				div_sxz[dim](self.dsx, self.dsz, self.sxx, self.szz, self.sxz, dx, dz, nx, nz)

				if spin:
					div_sy[dim](self.dsy_c, self.syx_c, self.syz_c, dx, dz, nx, nz)
					div_sy_c[dim](self.dsx, self.dsz, self.dsy_c, self.syy_c, self.j, dx, dz, nx, nz)

				stf_dsxz[self.nsrc, 1](self.dsx, self.dsz, self.stf_x, self.stf_z, self.src_id, isrc, it, nt)
				add_vxz[dim](self.vx, self.vz, self.ux, self.uz, self.dsx, self.dsz, self.rho, self.bound, dt)
				div_vxz[dim](self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.vx, self.vz, dx, dz, nx, nz)
				add_sxz[dim](self.sxx, self.szz, self.sxz, self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.lam, self.mu, dt)

				if spin:
					add_vy[dim](self.vy_c, self.uy_c, self.dsy_c, self.rho, self.bound, dt, npt)
					div_vy[dim](self.dvydx_c, self.dvydz_c, self.vy_c, dx, dz, nx, nz)
					add_sy_c[dim](self.syx_c, self.syy_c, self.syz_c, self.vy_c, self.dvydx_c, self.dvydz_c, self.dvxdz, self.dvzdx, self.dvzdz, self.nu, self.j, self.mu_c, self.nu_c, dt)

				save_obs[self.nrec, 1](self.obs_x, self.ux, self.rec_id, it, nt, nx, nz)
				save_obs[self.nrec, 1](self.obs_z, self.uz, self.rec_id, it, nt, nx, nz)

				if spin:
					save_obs[self.nrec, 1](self.obs_y, self.vy_c, self.rec_id, it, nt, nx, nz)

			if isa >= 0:
				if sh:
					self.vy.copy_to_host(self.vy_fwd[isa], stream=stream)

				if psv:
					self.vx.copy_to_host(self.vx_fwd[isa], stream=stream)
					self.vz.copy_to_host(self.vz_fwd[isa], stream=stream)

			if sfe and it > 0 and it % sfe == 0:
				if sh:
					self.vy.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'vy', it)

				if psv:
					self.vx.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'vx', it)
					self.vz.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'vz', it)

					if spin:
						self.vy_c.copy_to_host(out, stream=stream)
						stream.synchronize()
						self.export_field(out, 'ry', it)

		if 'output_traces' in self.path:
			tracedir = self.path['output_traces']
			nrec = self.nrec
			i = isrc if isrc >= 0 else 0

			if sh:
				out = np.zeros(nt * nrec, dtype='float32')
				self.obs_y.copy_to_host(out, stream=stream)
				np.save('%s/uy_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

			if psv:
				out = np.zeros(nt * nrec, dtype='float32')
				self.obs_x.copy_to_host(out, stream=stream)
				np.save('%s/ux_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

				out = np.zeros(nt * nrec, dtype='float32')
				self.obs_z.copy_to_host(out, stream=stream)
				np.save('%s/uz_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

				if spin:
					out = np.zeros(nt * nrec, dtype='float32')
					self.obs_y.copy_to_host(out, stream=stream)
					np.save('%s/ry_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

			stream.synchronize()

	def run_adjoint(self):
		stream = self.stream
		dim = self.dim
		sh = self.sh
		psv = self.psv

		nx = self.nx
		nz = self.nz
		nt = self.nt
		dx = self.dx
		dz = self.dz
		dt = self.dt

		npt = self.npt[0]
		sae = self.sae

		self.clear_wavefields()

		for it in range(self.nt):
			if sh:
				div_sy[dim](self.dsy, self.sxy, self.szy, dx, dz, nx, nz)
				stf_dsy[self.nrec, 1](self.dsy, self.adstf_y, self.rec_id, -1, it, nt)
				add_vy[dim](self.vy, self.uy, self.dsy, self.rho, self.bound, dt, npt)
				div_vy[dim](self.dvydx, self.dvydz, self.vy, dx, dz, nx, nz)
				add_sy[dim](self.sxy, self.szy, self.dvydx, self.dvydz, self.mu, dt, npt)
				if (it + sae) % sae == 0:
					isae = int((it + sae) / sae - 1)
					ndt = sae * dt
					self.dsy.copy_to_device(self.uy_fwd[isae], stream=stream)
					div_vy[dim](self.dvydx, self.dvydz, self.uy, dx, dz, nx, nz)
					div_vy[dim](self.dvydx_fw, self.dvydz_fw, self.dsy, dx, dz, nx, nz)
					interaction_muy[dim](self.k_mu, self.dvydx, self.dvydx_fw, self.dvydz, self.dvydz_fw, ndt, nx, nz)


			if psv:
				div_sxz[dim](self.dsx, self.dsz, self.sxx, self.szz, self.sxz, dx, dz, nx, nz)
				stf_dsxz[self.nsrc, 1](self.dsx, self.dsz, self.stf_x, self.stf_z, self.src_id, isrc, it, nt)
				add_vxz[dim](self.vx, self.vz, self.ux, self.uz, self.dsx, self.dsz, rho, bound, dt)
				div_vxz[dim](self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.vx, self.vz, dx, dz, nx, nz)
				add_sxz[dim](self.sxx, self.szz, self.sxz, self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.lam, self.mu, dt)

	def smooth(self, data):
		dim = self.dim
		apply_gauxxian_x[dim](data, self.gtmp, self.sigma, self.nx, self.nz)
		apply_gauxxian_z[dim](data, self.gtmp, self.gsum, self.sigma, self.nx, self.nz)

	def import_traces(self):
		nsrc = self.nsrc
		sh = self.sh
		psv = self.psv

		syn_x = self.syn_x = []
		syn_y = self.syn_y = []
		syn_z = self.syn_z = []

		if 'traces' in self.path:
			tracedir = self.path['traces']

			for i in range(nsrc):
				if self.mpi:
					if self.mpi.rank() != i:
						continue

				if sh:
					syn_y.append(np.fromfile('%s/vy_%06d.npy' % (tracedir, i), dtype='float32'))

				if psv:
					syn_x.append(np.fromfile('%s/vx_%06d.npy' % (tracedir, i), dtype='float32'))
					syn_z.append(np.fromfile('%s/vz_%06d.npy' % (tracedir, i), dtype='float32'))

		else:
			stream = self.stream
			nrec = self.nrec
			nt = self.nt

			tracedir = self.path['output'] + '/traces'

			if not self.mpi or self.mpi.rank() == 0:
				print('Generating traces')
				if not path.exists(tracedir):
					makedirs(tracedir)

			start = time()
			for i in range(nsrc):
				if self.mpi:
					if self.mpi.rank() != i:
						continue

				if not self.mpi:
					print('  task %02d / %02d' % (i+1, nsrc))

				self.taskid = i
				self.run_forward()

				if sh:
					out = np.zeros(nt * nrec, dtype='float32')
					self.obs_y.copy_to_host(out, stream=stream)
					syn_y.append(out)
					out.tofile('%s/vy_%06d.npy' % (tracedir, i))

				if psv:
					out = np.zeros(nt * nrec, dtype='float32')
					self.obs_x.copy_to_host(out, stream=stream)
					syn_x.append(out)
					out.tofile('%s/vx_%06d.npy' % (tracedir, i))

					out = np.zeros(nt * nrec, dtype='float32')
					self.obs_z.copy_to_host(out, stream=stream)
					syn_z.append(out)
					out.tofile('%s/vz_%06d.npy' % (tracedir, i))

				stream.synchronize()

			if not self.mpi:
				print('Elapsed time: %.2fs' % (time() - start))
				print('')

	def compute_misfit(self):
		return self.run_kernel(0)

	def compute_gradient(self):
		misfit, kernel, model =  self.run_kernel(1)
		npt = self.nx * self.nz
		out_k = np.zeros(npt, dtype='float32')
		out_m = np.zeros(npt, dtype='float32')
		kernel.copy_to_host(out_k, stream=self.stream)
		model.copy_to_host(out_m, stream=self.stream)
		self.stream.synchronize()

		return out_k, misfit, out_m

	def run_kernel(self, adj):
		if not self.mpi:
			if adj:
				print('Computing kernels')
			else:
				print('Computing misfit')

		stream = self.stream
		nsrc = self.nsrc
		nrec = self.nrec
		nt = self.nt

		sh = self.sh
		psv = self.psv

		self.setup_adjoint()
		self.clear_kernels()

		misfit=0

		for i in range(nsrc):
			if self.mpi and self.mpi.rank() != i:
				continue

			if not self.mpi:
				print('  task %02d / %02d' % (i+1, nsrc))

			self.taskid = i
			self.run_forward()

			j = 0 if self.mpi else i

			if sh:
				misfit += self._compute_misfit('y', self.syn_y[j])
				if adj:
					self.run_adjoint()

			if psv:
				misfit += self._compute_misfit('x', self.syn_x[j])
				misfit += self._compute_misfit('z', self.syn_z[j])
				if adj:
					self.run_adjoint()

		if self.mpi:
			mf_sum = np.zeros(1, dtype='float32')
			self.mpi.sum(misfit.astype('float32'), mf_sum)
			misfit = mf_sum[0]

			gradient = np.zeros(self.nx*self.nz,dtype='float32')
			self.k_mu.copy_to_host(gradient,stream=stream)
			stream.synchronize()

			rank = self.mpi.rank()
			fname = self.path['output'] + '/tmp/mu_' + str(rank) + '.npy'
			gradient.tofile(fname)

			self.mpi.sync()
			for i in range(self.nsrc):
				fname = self.path['output'] + '/tmp/mu_' + str(i) + '.npy'
				gradient += np.fromfile(fname, dtype='float32')

			self.k_mu = cuda.to_device(gradient, stream=stream)

		if adj:
			self.smooth(self.k_mu)

		if not self.mpi:
			print('  misfit = %.2f' % misfit)

		if adj:
			return misfit, self.k_mu, self.mu
		else:
			return misfit

	def update_model(self, mu):
		self.mu = cuda.to_device(mu, stream=self.stream)
