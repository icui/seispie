import numpy as np
import math

from os import path, makedirs
from time import time

from seispie.solver.base import base
from seispie.solver.source.ricker import ricker
from seispie.solver.misfit.waveform import waveform

from .solver import cuda, idx, idxij, run_forward, run_adjoint, run_kernel


@cuda.jit
def clear_field(field):
    k = idx()
    if k < field.size:
        field[k] = 0


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

class fd2d(base):
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
        self.rho = cuda.to_device(model['rho'], stream=stream)

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
            
            if self.spin:
                clear_field[dim](self.vy_c)
                clear_field[dim](self.uy_c)
                clear_field[dim](self.syx_c)
                clear_field[dim](self.syy_c)
                clear_field[dim](self.syz_c)

    def run_forward(self):
        run_forward(self)

    def run_adjoint(self):
        run_adjoint(self)

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
        return run_kernel(self, adj)

    def update_model(self, mu):
        self.mu = cuda.to_device(mu, stream=self.stream)
