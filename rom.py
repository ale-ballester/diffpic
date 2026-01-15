import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import json

import lineax as lx

def fom_state_to_rom_state(rho, mom, ene, K):
    """
    We ignore DC
    """
    rho_k = jnp.fft.rfft(rho)[1:K+1]
    mom_k = jnp.fft.rfft(mom)[1:K+1]
    ene_k = jnp.fft.rfft(ene)[1:K+1]
    x = jnp.concatenate([
        jnp.real(rho_k), jnp.real(mom_k), jnp.real(ene_k), 
        jnp.imag(rho_k), jnp.imag(mom_k), jnp.imag(ene_k)])
    return x

def rom_state_to_fom_state(y_rom, n0, p0, e0, *, N_mesh):
    """
    y: (3*2K,) with [Re(modes 1..K), Im(modes 1..K)]
    Returns rho,momentum,energy: (N_mesh,)
    Assumes DC modes are 0 and that there are three moments (density, momentum, energy)
    """
    K23 = y_rom.shape
    K = K23 // 6
    re = y_rom[:3*K]
    im = y_rom[3*K:2*3*K]
    c = re + 1j * im

    rho = jnp.zeros((N_mesh//2+1), dtype=c.dtype)
    momentum = jnp.zeros((N_mesh//2+1), dtype=c.dtype)
    energy = jnp.zeros((N_mesh//2+1), dtype=c.dtype)

    # DC
    rho = rho.at[:, 0].set(n0)
    momentum = momentum.at[:, 0].set(p0)
    energy = energy.at[:, 0].set(e0)

    # Positive modes: 1..K
    rho = rho.at[:, 1:K+1].set(c[:K])
    momentum = momentum.at[:, 1:K+1].set(c[K:2*K])
    energy = energy.at[:, 1:K+1].set(c[2*K:])

    rho = jnp.fft.irfft(rho).real
    momentum = jnp.fft.irfft(momentum).real
    energy = jnp.fft.irfft(energy).real

    return rho, momentum, energy

def square_lowpass(y, n0, K):
    """
    y: (T, Nx) real density on grid.
    Low-pass by keeping modes 1..K and their conjugate counterparts; DC removed.
    """
    y = y - jnp.mean(y, axis=-1, keepdims=True)
    y_k = jnp.fft.rfft(y, axis=-1)                # (T, Nx//2+1) complex

    y_k_trunc = jnp.zeros_like(y_k)

    # DC = 0
    y_k_trunc = y_k_trunc.at[:, 0].set(0.0)

    # Keep positive modes 1..K
    y_k_trunc = y_k_trunc.at[:, 1:K+1].set(y_k[:, 1:K+1])

    y_lp = jnp.fft.irfft(y_k_trunc, axis=-1).real
    return n0 + y_lp

def fit_discrete_linear_system_ridge(data, u, lam=1e-6, use_mean_center=False):
    """
    Fit x_{t+1} = A x_t + B u_t by ridge regression over all (batch,time) samples.

    Args:
        data: (B, T, nx) state trajectories
        u:    (B, T, nu) input trajectories
        lam: ridge regularization (scalar)
        use_mean_center: if True, mean-center x and u over all samples (often helps)

    Returns:
        A: (nx, nx)
        B: (nx, nu)
    """
    data = jnp.asarray(data)
    u = jnp.asarray(u)

    Bsz, T, nx = data.shape
    Bszu, Tu, nu = u.shape
    assert Tu == T, "u and data must have same T"
    assert Bszu == Bsz, "u and data must have same bs"

    # Build training pairs (x_t, u_t) -> x_{t+1}
    X  = data[:, :-1, :]      # (B, T-1, nx)
    Xp = data[:,  1:, :]      # (B, T-1, nx)
    U  = u[:, :-1, :]         # (B, T-1, nu)

    # Flatten batch+time into one big dataset of N = B*(T-1) samples
    N = Bsz * (T - 1)
    X  = X.reshape(N, nx)     # (N, nx)
    Xp = Xp.reshape(N, nx)    # (N, nx)
    U  = U.reshape(N, nu)     # (N, nu)

    if use_mean_center:
        X_mean  = jnp.mean(X, axis=0, keepdims=True)
        U_mean  = jnp.mean(U, axis=0, keepdims=True)
        Xp_mean = jnp.mean(Xp, axis=0, keepdims=True)
        X  = X  - X_mean
        U  = U  - U_mean
        Xp = Xp - Xp_mean

    # Z = [X, U]  (N, nx+nu)
    Z = jnp.concatenate([X, U], axis=1)  # (N, d)
    print(Z.shape)
    print(Xp.shape)
    d = nx + nu

    Z = lx.MatrixLinearOperator(Z)
    Theta = jax.vmap(lx.linear_solve, in_axes=(None, 1))(Z,Xp,solver=lx.AutoLinearSolver(well_posed=None)).value

    print(Theta.shape)

    A = Theta[:, :nx]   # (nx, nx)
    Bm = Theta[:, nx:]  # (nx, nu)
    return A, Bm


def rollout_discrete(A, B, x0, u):
    """
    Roll out x_{t+1} = A x_t + B u_t for t=0..T-2
    Args:
        A: (nx, nx)
        B: (nx, nu)
        x0: (nx,)
        u: (T, nu) or (T-1, nu). If (T,nu), last row is ignored.
    Returns:
        xs: (T, nx)
    """
    u = jnp.asarray(u)
    T = u.shape[0]
    nx = x0.shape[0]

    def step(x, ut):
        x_next = A @ x + B @ ut
        return x_next, x_next

    # Use u[:-1] so output length matches your data length when u is (T,nu)
    us = u[:-1] if T > 0 else u
    _, xs_body = jax.lax.scan(step, x0, us)
    xs = jnp.concatenate([x0[None, :], xs_body], axis=0)
    return xs

def fit_ct_trapezoid_ridge(X, U, dt, lam=1e-6):
    """
    Fit continuous-time linear system:
        dx/dt = A x + B u

    using trapezoidal rule over time and batching over trajectories.

    Parameters
    ----------
    X : (B, T, nx)
        State trajectories
    U : (B, T, nu)
        Input trajectories
    dt : float
        Time step
    lam : float
        Ridge regularization coefficient

    Returns
    -------
    A : (nx, nx)
    B : (nx, nu)
    """

    Bsz, T, nx = X.shape
    nu = U.shape[2]

    # --- Trapezoidal discretization ---
    # x_{k+1} - x_k = dt * (A * x_mid + B * u_mid)

    X_mid = 0.5 * (X[:, 1:, :] + X[:, :-1, :])   # (B, T-1, nx)
    U_mid = 0.5 * (U[:, 1:, :] + U[:, :-1, :])   # (B, T-1, nu)
    dX = (X[:, 1:, :] - X[:, :-1, :]) / dt       # (B, T-1, nx)

    # Flatten batch + time
    Z = jnp.concatenate(
        [X_mid.reshape(-1, nx), U_mid.reshape(-1, nu)],
        axis=1
    )  # (B*(T-1), nx+nu)

    Y = dX.reshape(-1, nx)  # (B*(T-1), nx)

    # Ridge regression solve:  (ZᵀZ + λI)Θ = ZᵀY
    d = nx + nu
    G = Z.T @ Z + lam * jnp.eye(d)
    RHS = Z.T @ Y

    Theta = jnp.linalg.solve(G, RHS)  # (d, nx)

    A = Theta[:nx, :].T
    B = Theta[nx:, :].T

    return A, B


class LDSModel(eqx.Module):
    A: jax.Array
    B: jax.Array
    dt: float = eqx.field(static=True)
    dim_in: int = eqx.field(static=True)
    dim_out: int = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    def __init__(self, dim_in, dim_out, dt, *, A=None, B=None, closed_loop=False, key):
        super().__init__()
        self.dt = dt
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.closed_loop = closed_loop

        key1, key2 = jax.random.split(key, num=2)
        if A is None:
            self.A = jnp.zeros((self.dim_out,self.dim_out))
        else:
            self.A = A
        if B is None:
            self.B = jnp.zeros((self.dim_out,self.dim_in))
        else:
            self.B = B

    def __call__(self, ts, y0, u):
        if self.closed_loop: # u is K matrix
            vf = lambda t,y,args: self.A(y) - u @ y
        else: # u is a signal
            control_interp = diffrax.LinearInterpolation(ts=ts, ys=u)
            vf = lambda t,y,args: self.A@y + self.B@control_interp.evaluate(t)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6, jump_ts=ts),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            hyperparams = {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "dt": self.dt,
            }
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
    
    @classmethod
    def load_model(cls, filename):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            dim_in   = hyperparams["dim_in"]
            dim_out   = hyperparams["dim_out"]
            dt    = hyperparams["dt"]

            # build skeleton with identical hyperparams
            model = cls(
                dim_in=dim_in,
                dim_out=dim_out,
                dt=dt,
                key=jax.random.PRNGKey(0),
            )

            # load parameters into that structure
            return eqx.tree_deserialise_leaves(f, model)