import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import scipy
from typing import Optional, Tuple

def _project_time_rfft_real_constraints(c: jax.Array, Nt: int) -> jax.Array:
    """Enforce the minimal constraints on an rFFT coefficient vector so irfft returns a real signal."""
    c = c.at[0].set(jnp.real(c[0]))
    if Nt % 2 == 0:
        c = c.at[Nt // 2].set(jnp.real(c[Nt // 2]))
    return c


def _rfft_truncated_time_signal(
    coeff_train: jax.Array, Nt: int, n_modes_time: int
) -> jax.Array:
    """
    coeff_train: (n_modes_time,) complex trainable prefix of the time rFFT coefficients.
    Returns: (Nt,) real time signal, with all higher time rFFT modes assumed zero.
    """
    Nt_pos = Nt // 2 + 1
    n_keep = int(min(n_modes_time, Nt_pos))
    full = jnp.zeros((Nt_pos,), dtype=coeff_train.dtype)
    full = full.at[:n_keep].set(coeff_train[:n_keep])
    full = _project_time_rfft_real_constraints(full, Nt)
    return jnp.fft.irfft(full, n=Nt)  # (Nt,), real


class FourierActuator(eqx.Module):
    # Trainable parameters (independent DOFs only)
    # a_hat_train[m, kt] and b_hat_train[m, kt] are *truncated* time-rFFT coefficients for each spatial mode m
    # Shapes:
    #   a_hat_train: (n_modes_space, n_modes_time) complex
    #   b_hat_train: (n_modes_space, n_modes_time) complex, but row m=0 is always zero (no sin at DC space mode)
    a_hat_train: jax.Array
    b_hat_train: jax.Array

    # Hyperparams / configuration
    zero: bool = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    Nt: int = eqx.field(static=True)
    N_mesh: int = eqx.field(static=True)
    boxsize: float = eqx.field(static=True)

    n_modes_time: int = eqx.field(static=True)   # how many time rFFT bins are trainable (prefix)
    n_modes_space: int = eqx.field(static=True)  # how many spatial Fourier modes (m=0..n_modes_space-1)

    # Optional closed-loop controller params (kept for compatibility with your design)
    K0: Optional[jax.Array] = eqx.field(static=True, default=None)
    u_max: Optional[jax.Array] = eqx.field(static=True, default=None)

    # Optional init metadata (not used in forward; useful to store)
    init_scale: float = eqx.field(static=True, default=0.0)

    def __init__(
        self,
        Nt: int,
        N_mesh: int,
        boxsize: float,
        *,
        n_modes_time: int,
        n_modes_space: int,
        key: Optional[jax.random.PRNGKey] = None,
        init_scale: float = 1e-4,
        zero: bool = False,
        closed_loop: bool = False,
        K0: Optional[jax.Array] = None,
        u_max: Optional[jax.Array] = None,
    ):
        self.Nt = int(Nt)
        self.N_mesh = int(N_mesh)
        self.boxsize = float(boxsize)

        self.n_modes_time = int(n_modes_time)
        self.n_modes_space = int(n_modes_space)

        self.zero = bool(zero)
        self.closed_loop = bool(closed_loop)
        self.K0 = K0
        self.u_max = u_max
        self.init_scale = float(init_scale)

        # Allocate trainable truncated time-rFFT coefficients
        shape = (self.n_modes_space, self.n_modes_time)

        if key is None:
            # deterministic (all zeros) if no key supplied
            a = jnp.zeros(shape, dtype=jnp.complex64)
            b = jnp.zeros(shape, dtype=jnp.complex64)
        else:
            k1, k2, k3, k4 = jax.random.split(key, num=4)
            # small random complex init
            a = init_scale * (
                jax.random.normal(k1, shape, dtype=jnp.float32)
                + 1j * jax.random.normal(k2, shape, dtype=jnp.float32)
            )
            b = init_scale * (
                jax.random.normal(k3, shape, dtype=jnp.float32)
                + 1j * jax.random.normal(k4, shape, dtype=jnp.float32)
            )
            a = a.astype(jnp.complex64)
            b = b.astype(jnp.complex64)

        # Enforce "no sin at m=0" (independent DOFs only)
        b = b.at[0].set(jnp.zeros((self.n_modes_time,), dtype=jnp.complex64))

        # Enforce time rFFT constraints on the *trainable prefix* where applicable:
        # - DC bin (kt=0) must be real
        a = a.at[:, 0].set(jnp.real(a[:, 0]))
        b = b.at[:, 0].set(jnp.real(b[:, 0]))
        # - Nyquist bin is only present if Nt even AND we are training it (kt == Nt//2 is within prefix)
        nyq = self.Nt // 2
        if self.Nt % 2 == 0 and self.n_modes_time > nyq:
            a = a.at[:, nyq].set(jnp.real(a[:, nyq]))
            b = b.at[:, nyq].set(jnp.real(b[:, nyq]))

        self.a_hat_train = a
        self.b_hat_train = b

    def field(self) -> jax.Array:
        """Return E(t, x) with shape (Nt, N_mesh), real."""
        # Spatial grid
        x = jnp.linspace(0.0, self.boxsize, self.N_mesh, endpoint=False)  # (N_mesh,)
        m = jnp.arange(self.n_modes_space)                                # (n_modes_space,)
        k = 2.0 * jnp.pi * m / self.boxsize                               # (n_modes_space,)

        cos_kx = jnp.cos(k[None, :] * x[:, None])                         # (N_mesh, n_modes_space)
        sin_kx = jnp.sin(k[None, :] * x[:, None])                         # (N_mesh, n_modes_space)
        sin_kx = sin_kx.at[:, 0].set(0.0)                                 # enforce no sin at m=0

        # Build time signals a_m(t), b_m(t) from truncated time rFFT coeffs
        # vmap over spatial mode index m
        a_t = jax.vmap(lambda c: _rfft_truncated_time_signal(c, self.Nt, self.n_modes_time))(self.a_hat_train)
        b_t = jax.vmap(lambda c: _rfft_truncated_time_signal(c, self.Nt, self.n_modes_time))(self.b_hat_train)
        # a_t, b_t: (n_modes_space, Nt) -> (Nt, n_modes_space)
        a_t = jnp.swapaxes(a_t, 0, 1)
        b_t = jnp.swapaxes(b_t, 0, 1)

        # Combine time amplitudes with spatial basis
        E = (a_t @ cos_kx.T) + (b_t @ sin_kx.T)                           # (Nt, N_mesh)
        return E  # real

    def __call__(self, n, x=None):
        """Return E[n] (N_mesh,) for open-loop, or closed-loop control if enabled."""
        if self.zero:
            return jnp.zeros(self.N_mesh)

        if self.closed_loop:
            if x is None:
                raise ValueError("closed_loop=True requires state x to be provided.")
            x_rom = x  # replace with ROM mapping
            u = -(self.K0 @ x_rom)
            u = self.u_max * jnp.tanh(u / self.u_max)
            return u

        # Open-loop: precompute full field and slice
        E_all = self.field()
        return E_all[n.astype(int)]
    
    def get_modes_summary(self):
        """
        Returns paper-style coefficients for the *static* part of the field:
            E(x) = c0 + sum_{m>=1} [ c_m cos(k_m x) + s_m sin(k_m x) ]

        Only meaningful when n_modes_time == 1 (time-DC).
        """
        if self.n_modes_time != 1:
            #raise ValueError("get_modes_summary is only valid for n_modes_time == 1 (static field).")
            return ""

        # Time-DC amplitudes (scalars)
        # a_hat_train[m, 0] and b_hat_train[m, 0] are real by construction
        a0 = jnp.real(self.a_hat_train[:, 0])  # (n_modes_space,)
        b0 = jnp.real(self.b_hat_train[:, 0])  # (n_modes_space,)

        summary = []
        for m in range(self.n_modes_space):
            k_m = 2 * jnp.pi * m / self.boxsize
            if m == 0:
                summary.append({
                    "m": 0,
                    "k": 0.0,
                    "offset": float(a0[0]),
                })
            else:
                summary.append({
                    "m": m,
                    "k": float(k_m),
                    "cos_coeff": float(a0[m]),
                    "sin_coeff": float(b0[m]),
                })
        return summary

    # -----------------------
    # Save / Load
    # -----------------------
    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": self.zero,
                "closed_loop": self.closed_loop,
                "Nt": self.Nt,
                "N_mesh": self.N_mesh,
                "boxsize": self.boxsize,
                "n_modes_time": self.n_modes_time,
                "n_modes_space": self.n_modes_space,
                "init_scale": self.init_scale,
                # closed-loop extras (may be None)
                "has_K0": self.K0 is not None,
                "has_u_max": self.u_max is not None,
            }
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            model = cls(
                Nt=hyperparams["Nt"],
                N_mesh=hyperparams["N_mesh"],
                boxsize=hyperparams["boxsize"],
                n_modes_time=hyperparams["n_modes_time"],
                n_modes_space=hyperparams["n_modes_space"],
                key=None,  # we'll overwrite parameters from file
                init_scale=hyperparams.get("init_scale", 0.0),
                zero=hyperparams["zero"],
                closed_loop=hyperparams["closed_loop"],
                K0=None,     # overwritten if present in leaves
                u_max=None,  # overwritten if present in leaves
            )

            return eqx.tree_deserialise_leaves(f, model)

class ModeFeedbackActuator(eqx.Module):
    # ---- required geometry ----
    N_mesh: int
    boxsize: float

    # ---- feedback config ----
    mlp: eqx.Module
    K0: jax.Array              # control
    dc_value: jax.Array
    n_modes_space_in: int = 4     # Number of modes to control
    n_modes_space_out: int = 4     # Number of modes to control
    init_scale: float = 1.0    # For initialization

    width: int = 64
    depth: int = 2

    # ---- constraints / regularization knobs ----
    use_linear: bool = False
    include_dc: bool = False        # usually False on periodic domain
    u_max: float | None = None      # optional clip on |u_m| in Fourier domain

    # ---- compatibility flags ----
    zero: bool = False              # if True, always return 0 field
    closed_loop: bool = True        # required so PIC knows to pass measurements, False not implemented

    def __init__(self,N_mesh,boxsize,use_linear=False,width=64,depth=2,include_dc=False,u_max=None,zero=False,closed_loop=True,n_modes_space_in=4,n_modes_space_out=4,init_scale=1.0,*,key):
        self.N_mesh = N_mesh
        self.boxsize = boxsize
        self.include_dc = include_dc
        self.u_max = u_max
        self.zero = zero
        self.closed_loop = closed_loop
        self.n_modes_space_in = n_modes_space_in
        self.n_modes_space_out = n_modes_space_out
        self.init_scale = init_scale
        self.use_linear = use_linear

        if self.use_linear:
            k1, k2, k3 = jax.random.split(key, num=3)
            shape = (self.n_modes_space_out,self.n_modes_space_in)
            self.K0 = self.init_scale * (
                    jax.random.normal(k1, shape, dtype=jnp.float64)
                    + 1j * jax.random.normal(k2, shape, dtype=jnp.float64)
                )
            self.K0 = self.K0.astype(jnp.complex64)
            if self.include_dc:
                self.dc_value = self.init_scale * jax.random.normal(k3, (1,), dtype=jnp.float64)
            else:
                self.dc_value = None
            self.mlp = None
        else:
            self.K0 = None
            self.dc_value = None
            in_size = 2*self.n_modes_space_in
            out_size = 2*self.n_modes_space_out
            if self.include_dc: 
                in_size += 1
                out_size += 1
            self.mlp = eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=width,
                depth=depth,
                activation=jnn.tanh,
                key=key,
            )

    def __call__(self, n: int, *, state=None):
        """Return E_ext(x) on the grid for step index n.

        Parameters
        ----------
        n : int
            time-step index (kept for interface compatibility; not used here).
        state : complex array, shape (N_mesh//2+1,)
            One-sided rFFT coefficients of state(x) at current step.
\
        Returns
        -------
        E_ext : float array, shape (N_mesh,)
        """
        #jax.debug.print("Current gain: {gain}", gain=jnp.linalg.norm(self.K0))
        if self.zero:
            return jnp.zeros((self.N_mesh,), dtype=jnp.float32)

        if self.use_linear:
            meas = state[1:self.n_modes_space_in+1]

            # Feedback in Fourier domain: u_m = -K * meas
            u_m = (-self.K0) @ meas

            # Optional magnitude clipping (in complex plane)
            if self.u_max is not None:
                mag = jnp.abs(u_m)
                u_m = jnp.where(mag > self.u_max, u_m * (self.u_max / (mag + 1e-12)), u_m)

            # Build one-sided spectrum for E_ext and invert to real space
            spec = jnp.zeros((self.N_mesh // 2 + 1,), dtype=jnp.complex64)

            if self.include_dc:
                spec = spec.at[0].set(jnp.array(self.dc_value[0], dtype=jnp.complex64))

            spec = spec.at[1:self.n_modes_space_out+1].set(u_m.astype(jnp.complex64))
        else:
            state = state[:self.n_modes_space_in+1]
            if not self.include_dc: state = state[1:]
            u_m = self.mlp(jnp.concatenate((jnp.real(state),jnp.imag(state))))

            spec = jnp.zeros((self.N_mesh // 2 + 1,), dtype=jnp.complex64)

            if self.include_dc:
                spec = spec.at[0].set(u_m[0])
                modes = u_m[1:self.n_modes_space_out+1] + 1j * u_m[self.n_modes_space_out+1:]
                spec = spec.at[1:self.n_modes_space_out].set(modes)
            else:
                modes = u_m[:self.n_modes_space_out] + 1j * u_m[self.n_modes_space_out:]
                spec = spec.at[1:self.n_modes_space_out+1].set(modes)       

        E_ext = jnp.fft.irfft(spec, n=self.N_mesh).real  # -> real-valued (N_mesh,)
        #jax.debug.print("Output: {out}", out=jnp.linalg.norm(E_ext))
        return E_ext

    # -----------------------
    # Save / Load
    # -----------------------
    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": bool(self.zero),
                "closed_loop": bool(self.closed_loop),
                "N_mesh": int(self.N_mesh),
                "boxsize": float(self.boxsize),
                "width": int(self.width),
                "depth": int(self.depth),
                "n_modes_space_in": int(self.n_modes_space_in),
                "n_modes_space_out": int(self.n_modes_space_out),
                "init_scale": float(self.init_scale),

                # actuator-specific hyperparams
                "include_dc": bool(self.include_dc),
                "use_linear": bool(self.use_linear),
                "u_max": float(self.u_max) if self.u_max is not None else None
            }
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            model = cls(
                # schema-compatible fields
                N_mesh=hyperparams["N_mesh"],
                boxsize=hyperparams["boxsize"],
                n_modes_space_in=hyperparams.get("n_modes_space_in", 0),
                n_modes_space_out=hyperparams.get("n_modes_space_out", 0),
                width=hyperparams.get("width", 64),
                depth=hyperparams.get("depth", 2),
                init_scale=hyperparams.get("init_scale", 0.0),
                zero=hyperparams.get("zero", False),
                closed_loop=hyperparams.get("closed_loop", True),

                # actuator-specific fields
                include_dc=hyperparams.get("include_dc", False),
                use_linear=hyperparams.get("use_linear", False),

                # placeholders overwritten by leaves (if present)
                u_max=hyperparams.get("u_max", None),
                key=jax.random.key(0)
            )

            return eqx.tree_deserialise_leaves(f, model)

def ctrb(A, B):
    n = A.shape[0]
    blocks = []
    AB = B
    for _ in range(n):
        blocks.append(AB)
        AB = A @ AB
    return jnp.concatenate(blocks, axis=1)

def continuous_lqr(A, B, Q=None, R=None):
    """
    Continuous-time LQR for xdot = A x + B u.
    Returns K, P, eigvals(A-BK)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    P = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)

    # K = R^{-1} B^T P
    K = jnp.linalg.solve(R_np, B_np.T @ P)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)

def discrete_lqr(A, B, Q=None, R=None):
    """
    Discrete-time LQR for x_{k+1} = A x_k + B u_k.
    Minimizes sum_{k=0}^\infty (x_k^T Q x_k + u_k^T R u_k).
    Returns K, P, eigvals(A - B K)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve DARE: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    # scipy returns a NumPy array; we'll wrap back to jnp
    P = scipy.linalg.solve_discrete_are(
        jnp.asarray(A_np), jnp.asarray(B_np), jnp.asarray(Q_np), jnp.asarray(R_np)
    )

    P = jnp.asarray(P)

    # K = (R + B^T P B)^{-1} (B^T P A)
    S = R_np + B_np.T @ P @ B_np
    K = jnp.linalg.solve(S, B_np.T @ P @ A_np)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)