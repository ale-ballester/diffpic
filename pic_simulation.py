import jax
import jax.numpy as jnp
import equinox as eqx

class PICSimulation(eqx.Module):
    boxsize: float = eqx.field(static=True)
    N_particles: int = eqx.field(static=True)
    N_mesh: int = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    n0: float = eqx.field(static=True)
    vb: float = eqx.field(static=True)
    vth: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    n_steps: float = eqx.field(static=True)
    m: float = eqx.field(static=True)
    q: float = eqx.field(static=True)
    eps0: float = eqx.field(static=True)

    # Frequency
    k: jax.Array
    nonzero_k: jax.Array
    k_masked: jax.Array
    k_masked_inv2: jax.Array

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

    def __init__(self, boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, m=1, q=1, eps0=1, t0=0, higher_moments=False):
        self.boxsize = boxsize
        self.N_particles = N_particles
        self.N_mesh = N_mesh
        self.dx = self.boxsize / self.N_mesh
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

        # Frequencies
        self.k = 2 * jnp.pi * jnp.fft.fftfreq(self.N_mesh, d=self.dx)  # Wavenumbers
        self.nonzero_k = self.k != 0
        self.k_masked = jnp.where(self.nonzero_k, self.k, 1.0)
        self.k_masked_inv2 = 1.0/self.k_masked**2

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
    
    def create_y0(self, key, pos_sample: bool = False):
        key_pos, key_vel = jax.random.split(key, 2)

        N = self.N_particles
        L = self.boxsize

        if pos_sample:
            pos = jax.random.uniform(key_pos, (N, 1)) * L
        else:
            idx = jax.random.permutation(key_pos, N)
            pos = (idx.astype(jnp.float64) * (L / N))[:, None]

        vel = self.vth * jax.random.normal(key_vel, (N, 1)) + self.vb
        Nh = N // 2
        vel = vel.at[Nh:, :].set(-vel[Nh:, :])
        vel = vel - jnp.mean(vel)

        return (pos, vel)

    def cic_deposition(self, pos, vel=None):
        pos = jnp.mod(pos, self.boxsize)

        x = pos / self.dx
        j = jnp.floor(x).astype(jnp.int32)
        j = jnp.mod(j, self.N_mesh)
        jp1 = jnp.mod(j + 1, self.N_mesh)
        frac = x - j.astype(x.dtype)
        weight_j = 1.0 - frac
        weight_jp1 = frac
        w0 = self.q * self.n0 * (self.boxsize / self.N_particles) / self.dx

        def deposit(q=None):
            if q is None:
                g = jax.ops.segment_sum(weight_j[:, 0], j[:, 0], num_segments=self.N_mesh)
                g += jax.ops.segment_sum(weight_jp1[:, 0], jp1[:, 0], num_segments=self.N_mesh)
            else:
                g = jax.ops.segment_sum((weight_j * q)[:, 0], j[:, 0], num_segments=self.N_mesh)
                g += jax.ops.segment_sum((weight_jp1 * q)[:, 0], jp1[:, 0], num_segments=self.N_mesh)
            return g * w0
    
        moments = deposit()[:,None]
        if self.higher_moments:
            momentum = deposit(self.m * vel)[:,None]
            energy = deposit(0.5 * self.m * vel**2)[:,None]
            moments = jnp.concatenate((moments,momentum,energy),axis=-1)
        return moments, j, jp1, weight_j, weight_jp1

    def poisson_solver(self, rho):
        rho_k = jnp.fft.fft(rho)
        rho_k = rho_k.at[0].set(0) # Enforce quasineutrality
        phi_k = jnp.where(self.nonzero_k, -rho_k*self.k_masked_inv2, 0.0)
        E_k = -1j * self.k * phi_k / self.eps0  # Electric field in k-space
        E = jnp.fft.ifft(E_k).real  # Electric field in real space
        return E, rho_k

    def cic_gather(self, y, E_grid, j, jp1, weight_j, weight_jp1, E_ext=None):
        pos, vel, acc = y
        # Interpolate grid value onto particle locations
        E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

        # Add external electric field
        if E_ext is not None:
            E += weight_j * E_ext[j] + weight_jp1 * E_ext[jp1]

        return E

    def step(self, y, n, E_control=None):
        pos, vel, acc, E_field, E_ext, moments = y

        # (1/2) kick
        vel += acc * self.dt / 2.0

        # drift (and apply periodic boundary conditions)
        pos += vel * self.dt
        pos = jnp.mod(pos, self.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[:,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(n, state=jnp.fft.rfft(moments[:,0]))
            else:
                E_ext = E_control(n)

        E = self.cic_gather((pos, vel, acc), E_grid, j, jp1, weight_j, weight_jp1, E_ext=E_ext)

        # update accelerations
        acc = -self.q*E/self.m

        # (1/2) kick
        vel += acc * self.dt / 2.0

        return pos, vel, acc, E_grid, E_ext, moments

    def run_simulation(self, y0, E_control=None):
        pos, vel = y0

        pos = jnp.mod(pos, self.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[:,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(jnp.asarray(0), state=jnp.fft.rfft(moments[:,0]))
            else:
                E_ext = E_control(jnp.asarray(0))

        E = self.cic_gather((pos,vel,jnp.zeros_like(pos)), E_grid, j, jp1, weight_j, weight_jp1, E_ext=E_ext)

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
                (pos_traj.squeeze(), vel_traj.squeeze(), acc_traj.squeeze(), E_traj, Eext_traj, moments_traj[:,:,0], moments_traj[:,:,1], moments_traj[:,:,2]),
                is_leaf=lambda x: x is None,
            )
        else:
            new_obj = eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho),
                self,
                (pos_traj.squeeze(), vel_traj.squeeze(), acc_traj.squeeze(), E_traj, Eext_traj, moments_traj[:,:,0]),
                is_leaf=lambda x: x is None,
            )
        return new_obj