import jax
import math
import jax.numpy as jnp
import equinox as eqx
import itertools
from typing import Tuple

class PICSimulation(eqx.Module):
    dim: int = eqx.field(static=True)
    boxsize: jax.Array = eqx.field(static=True)
    N_particles: int = eqx.field(static=True)
    N_mesh: Tuple[int, ...] = eqx.field(static=True)
    Ng: int = eqx.field(static=True)
    dx: jax.Array = eqx.field(static=True)
    n0: float = eqx.field(static=True)
    vb: jax.Array = eqx.field(static=True)
    vth: jax.Array = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    n_steps: float = eqx.field(static=True)
    m: float = eqx.field(static=True)
    q: float = eqx.field(static=True)
    eps0: float = eqx.field(static=True)
    n_sort: int = eqx.field(static=True)

    # Frequency
    kvec: jax.Array
    k2: jax.Array
    nonzero_k: jax.Array

    # Trajectories
    ts: jax.Array
    positions: jax.Array
    velocities: jax.Array
    accelerations: jax.Array
    E_field: jax.Array
    E_ext: jax.Array
    rho: jax.Array
    higher_moments: bool = eqx.field(static=True)
    momentum: jax.Array
    energy: jax.Array

    def __init__(self, dim, boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, m=1, q=1, Z=1, eps0=1, t0=0, higher_moments=False, n_sort=None):
        self.dim = dim
        self.boxsize = boxsize
        self.N_particles = N_particles
        self.N_mesh = N_mesh
        self.Ng = math.prod(self.N_mesh)
        self.dx = self.boxsize / jnp.array(self.N_mesh)
        self.n0 = n0 # Background number density
        self.vb = vb
        self.vth = vth
        self.dt = dt
        self.t0 = t0
        self.t1 = t1
        self.n_steps = int(jnp.floor((self.t1-self.t0) / dt))
        self.m = m
        self.q = q
        self.eps0 = eps0
        self.n_sort = n_sort

        # Frequencies
        d = len(self.N_mesh)
        k1d = [
            (2 * jnp.pi) * jnp.fft.fftfreq(self.N_mesh[a], d=float(self.boxsize[a] / self.N_mesh[a]))
            for a in range(d)
        ]  # list of (Na,) arrays

        # Mesh the components onto full N_mesh grid
        kvec = jnp.meshgrid(*k1d, indexing="ij")  # tuple length d, each shape N_mesh
        self.kvec = kvec

        self.k2 = jnp.zeros(self.N_mesh)
        for kc in self.kvec:
            self.k2 = self.k2 + kc ** 2

        # Mask out k=0 (the DC mode). For a full fftn grid, DC is at index (0,0,...,0).
        self.nonzero_k = jnp.ones(self.N_mesh, dtype=bool).at[(0,) * d].set(False)

        # Trajectories
        self.ts = self.t0 + dt * jnp.arange(self.n_steps)
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.E_field = None
        self.E_ext = None
        self.rho = None
        self.higher_moments = higher_moments
        self.momentum = None
        self.energy = None
    
    def sort_particles(self,pos,vel):
        cell = jnp.floor(pos / self.dx).astype(jnp.int32)
        cell = jnp.mod(cell, jnp.array(self.N_mesh))
        cell_id = self._ravel_multi_index(cell, jnp.array(self.N_mesh))

        perm = jnp.argsort(cell_id)
        pos = pos[perm]
        vel = vel[perm]

        return pos, vel
    
    def maybe_sort(self,pos,vel,n):
        def do_sort(state):
            pos,vel = state
            return self.sort_particles(pos,vel)

        def no_sort(state):
            return pos,vel

        return jax.lax.cond(
            (n % self.n_sort) == 0,
            do_sort,
            no_sort,
            (pos,vel)
        )
    
    def create_y0(
        self,
        key,
    ):
        key_pos, key_vel = jax.random.split(key, 2)

        shape = (self.N_particles,self.dim)

        x = jax.random.uniform(key_pos,shape=shape,minval=jnp.zeros(self.dim),maxval=self.boxsize)
        pos = jnp.mod(x,self.boxsize)

        # -------------------------
        # Velocities
        # -------------------------
        # thermal
        vel = self.vth * jax.random.normal(key_vel, shape=(self.N_particles, self.dim)) + self.vb
        Nh = self.N_particles // 2
        vel = vel.at[Nh:, :].set(-vel[Nh:, :])

        # enforce zero total momentum (helps remove finite-N bias)
        vel = vel - jnp.mean(vel, axis=0, keepdims=True)

        if self.n_sort is not None:
            pos, vel = self.sort_particles(pos,vel)

        return (pos, vel)
    
    def _ravel_multi_index(self, idxs, dims):
        """idxs: (..., d) in [0, dims); dims: (d,) ints -> linear index (...,)"""
        dims = jnp.asarray(dims, dtype=jnp.int32)
        strides = jnp.concatenate([jnp.cumprod(dims[::-1])[::-1][1:], jnp.array([1], jnp.int32)])
        # strides = [prod(dims[1:]), prod(dims[2:]), ..., 1]
        return jnp.sum(idxs * strides, axis=-1)

    def cic_deposition(self, pos, vel=None):
        pos = jnp.mod(pos, self.boxsize)

        N_mesh = jnp.asarray(self.N_mesh, dtype=jnp.int32)
        x = pos / self.dx                                # (Np, d)  dx is (d,)
        j0 = jnp.floor(x).astype(jnp.int32)              # (Np, d)
        frac = x - j0.astype(x.dtype)                    # (Np, d)
        j0 = jnp.mod(j0, N_mesh)                    # periodic wrap
        d = frac.shape[-1]

        cell_vol = jnp.prod(self.dx)
        w0 = (self.n0) * (jnp.prod(self.boxsize) / self.N_particles) / cell_vol

        #grid_idx = [slice(0,1) for i in range(d)]
        #meshgrid = jnp.mgrid[grid_idx]

        def deposit(q=None):
            if q is None:
                shape = (*self.N_mesh, 1)
                moment_flat = jnp.zeros((self.Ng,1))
            else:
                shape = (*self.N_mesh, q.shape[-1])
                moment_flat = jnp.zeros((self.Ng,q.shape[-1]))
            for bits in itertools.product([0, 1], repeat=int(d)):
                b = jnp.array(bits, dtype=jnp.int32)                     # (d,)
                # node indices for this corner
                jj = jnp.mod(j0 + b, N_mesh)                             # (Np, d)
                lin = self._ravel_multi_index(jj, N_mesh)                # (Np,)

                # product weight for this corner
                w = jnp.where(b[None, :] == 1, frac, 1.0 - frac)         # (Np, d)
                w = jnp.prod(w, axis=-1)            # (Np,)
                
                if q is None: q=1
                contrib = w0[...,None] * w[...,None] * q
                moment_flat = moment_flat.at[lin].add(contrib)
            return moment_flat.reshape(shape)
            
        moments = deposit(jnp.array(self.q)[None]) # rho (charge density)
        if self.higher_moments:
            momentum = deposit(self.m * vel) # These are mass moments
            energy = deposit(0.5 * self.m * jnp.sum(vel**2,axis=-1,keepdims=True)) # TODO: These are mass moments
            moments = jnp.concatenate((moments,momentum,energy),axis=-1)
        return moments

    def cic_shape_factor(self):
        # returns S(k) with shape N_mesh
        S = 1.0
        for a in range(self.dim):
            ka = self.kvec[a]
            # sinc(x) in numpy/jax is sin(pi x)/(pi x), so we must scale argument accordingly
            # We want sinc(ka*dx/2) = sin(ka*dx/2)/(ka*dx/2)
            # => use jnp.sinc((ka*dx/2)/pi) = jnp.sinc(ka*dx/(2*pi))
            S = S * (jnp.sinc(ka * self.dx[a] / (2*jnp.pi))**2)
        return S

    def poisson_solver(self, rho):
        rho_k = jnp.fft.fftn(rho)
        rho_k = jnp.where(self.nonzero_k, rho_k, 0.0)

        S = self.cic_shape_factor()
        S2 = jnp.where(self.nonzero_k, S**2, 1.0)

        # Poisson: phi_k = - rho_k / (eps0 * k^2 * |S|^2)
        phi_k = jnp.where(self.nonzero_k, -rho_k / (self.eps0 * self.k2 * S2), 0.0)

        # E_k = - i k phi_k   (no /eps0 here)
        E_k = [(-1j * kcomp * phi_k) for kcomp in self.kvec]

        E = jnp.stack([jnp.fft.ifftn(Ek).real for Ek in E_k], axis=-1)
        return E, rho_k

    def all_corner_bits(self, d: int, dtype=jnp.int32):
        C = 1 << d
        c = jnp.arange(C, dtype=jnp.uint32)[:, None]          # (C, 1)
        shifts = jnp.arange(d, dtype=jnp.uint32)[None, :]     # (1, d)
        B = (c >> shifts) & jnp.uint32(1)                     # (C, d)
        return B.astype(dtype)

    def cic_gather(self, y, E_grid, E_ext=None):
        """
        y: (pos, vel, acc) where pos is (Np, d)
        E_grid: (*N_mesh, d) real
        E_ext:  (*N_mesh, d) real or None
        Returns:
        E: (Np, d) gathered field at particles
        """
        pos, vel, acc = y
        pos = jnp.mod(pos, self.boxsize)                      # (Np, d)

        x = pos / self.dx                                     # (Np, d)
        j0 = jnp.floor(x).astype(jnp.int32)                   # (Np, d)
        frac = x - j0.astype(x.dtype)                         # (Np, d)
        N_mesh = jnp.asarray(self.N_mesh, dtype=jnp.int32)
        j0 = jnp.mod(j0, N_mesh)                              # periodic wrap

        Np, d = frac.shape
        B = self.all_corner_bits(d, dtype=jnp.int32)               # (C, d)
        C = B.shape[0]

        # Corner indices for each particle: (C, Np, d)
        jj = jnp.mod(j0[None, :, :] + B[:, None, :], N_mesh[None, None, :])

        # Corner weights: (C, Np)
        w = jnp.where(B[:, None, :] == 1, frac[None, :, :], 1.0 - frac[None, :, :])
        w = jnp.prod(w, axis=-1)          # (C, Np)

        # Gather E_grid at all corners.
        # Use advanced indexing by splitting indices per dimension.
        idx = tuple(jj[..., a] for a in range(d))             # tuple of (C, Np) int arrays
        Eg = E_grid[idx]                                      # (C, Np, d)

        if E_ext is not None:
            Eg = Eg + E_ext[idx]                              # (C, Np, d)

        # Weighted sum over corners -> (Np, d)
        E = jnp.sum(w[..., None] * Eg, axis=0)                # sum over C

        return E

    def step(self, y, n, E_control=None):
        pos, vel, acc, E_field, E_ext, moments = y

        # (1/2) kick
        vel += acc * self.dt / 2.0

        # drift (and apply periodic boundary conditions)
        pos += vel * self.dt
        pos = jnp.mod(pos, self.boxsize)

        moments = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[...,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(n, state=jnp.fft.rfft(moments[...,0]))
            else:
                E_ext = E_control(n)

        E = self.cic_gather((pos, vel, acc), E_grid, E_ext=E_ext)

        # update accelerations
        acc = -self.q*E/self.m

        # (1/2) kick
        vel += acc * self.dt / 2.0

        # Sort particles
        if self.n_sort is not None: pos,vel = self.maybe_sort(pos,vel,n)

        return pos, vel, acc, E_grid, E_ext, moments

    @jax.jit
    def run_simulation(self, y0, E_control=None):
        pos, vel = y0

        pos = jnp.mod(pos, self.boxsize)

        moments = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[...,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(jnp.asarray(0), state=jnp.fft.rfft(moments[...,0]))
            else:
                E_ext = E_control(jnp.asarray(0))

        E = self.cic_gather((pos,vel,jnp.zeros_like(pos)), E_grid, E_ext=E_ext)

        acc = -self.q*E/self.m

        y0 = (pos, vel, acc, E_grid, E_ext, moments)

        def step_fn(y, n):
            y_next = self.step(y, n, E_control=E_control)
            return y_next, y_next

        _, outs = jax.lax.scan(step_fn, y0, xs=jnp.arange(len(self.ts)), length=self.n_steps)

        pos_traj, vel_traj, acc_traj, E_traj, Eext_traj, moments_traj = outs

        new_obj = None
        if self.higher_moments:
            new_obj = eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho, s.momentum, s.energy),
                self,
                (pos_traj, vel_traj, acc_traj, E_traj, Eext_traj, moments_traj[...,0], moments_traj[...,1:3], moments_traj[...,3]),
                is_leaf=lambda x: x is None,
            )
        else:
            new_obj = eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho),
                self,
                (pos_traj, vel_traj, acc_traj, E_traj, Eext_traj, moments_traj[:,:,0]),
                is_leaf=lambda x: x is None,
            )
        return new_obj