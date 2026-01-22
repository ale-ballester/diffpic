import os
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML


# -----------------------------
# Utilities
# -----------------------------
def _maybe_save_or_show(fig, save_path=None, dpi=200):
    """If save_path is None -> show; else save and close."""
    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _extent_xy(boxsize):
    # imshow extent for (Nx,Ny) array with origin lower
    # x in [0,Lx], y in [0,Ly]
    Lx, Ly = float(boxsize[0]), float(boxsize[1])
    return [0.0, Lx, 0.0, Ly]


def _grid_axes(N_mesh, boxsize):
    Nx, Ny = int(N_mesh[0]), int(N_mesh[1])
    Lx, Ly = float(boxsize[0]), float(boxsize[1])
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    return x, y


def _field_energy(E_field, boxsize, N_mesh, eps0=1.0):
    """
    E_field: (..., Nx, Ny, 2) or (Nx,Ny,2)
    returns energy scalar (sum over grid) using cell area
    """
    # cell area
    dx = float(boxsize[0]) / int(N_mesh[0])
    dy = float(boxsize[1]) / int(N_mesh[1])
    dA = dx * dy
    Emag2 = jnp.sum(E_field**2, axis=-1)  # (..., Nx, Ny)
    return 0.5 * eps0 * jnp.sum(Emag2) * dA


def _kinetic_energy(energy):
    return jnp.sum(energy,axis=(0,1))

def _momentum(momentum):
    return jnp.sum(momentum,axis=(0,1))


def _mass_from_rho(rho, boxsize, N_mesh):
    # rho: (Nx,Ny)
    dx = float(boxsize[0]) / int(N_mesh[0])
    dy = float(boxsize[1]) / int(N_mesh[1])
    return jnp.sum(rho) * dx * dy


# -----------------------------
# A) Diagnostics snapshot plots
# -----------------------------
def plot_fields_snapshot(
    sim,
    t_index=-1,
    what=("rho", "E_mag", "Ex", "Ey"),
    clim=None,
    save_path=None,
    title_prefix="",
):
    """
    Plot 2D heatmaps for rho / E components / |E| at a given time index.
    """
    assert sim.dim == 2, "This snapshot routine is for dim=2."
    Nx, Ny = int(sim.N_mesh[0]), int(sim.N_mesh[1])

    fig, axes = plt.subplots(1, len(what), figsize=(4.5 * len(what), 4), squeeze=False)
    axes = axes[0]
    extent = _extent_xy(sim.boxsize)

    rho = None
    E = None
    if "rho" in what:
        rho = sim.rho[t_index]
        assert rho.shape == (Nx, Ny)
    if any(k in what for k in ["E_mag", "Ex", "Ey"]):
        E = sim.E_field[t_index]
        assert E.shape == (Nx, Ny, 2)

    for ax, key in zip(axes, what):
        if key == "rho":
            img = rho
            ttl = "rho"
        elif key == "E_mag":
            img = jnp.sqrt(jnp.sum(E**2, axis=-1))
            ttl = "|E|"
        elif key == "Ex":
            img = E[..., 0]
            ttl = "E_x"
        elif key == "Ey":
            img = E[..., 1]
            ttl = "E_y"
        else:
            raise ValueError(f"Unknown field key: {key}")

        im = ax.imshow(
            np.array(img),
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        if clim is not None and key in clim:
            im.set_clim(*clim[key])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}{ttl} @ t={float(sim.ts[t_index]):.3g}")
        fig.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)


def plot_lineouts(
    sim,
    t_index=-1,
    y_index=None,
    x_index=None,
    what=("rho", "Ex", "Ey", "E_mag"),
    save_path=None,
):
    """
    1D lineouts through the 2D fields at a given time:
      - along x at fixed y_index
      - along y at fixed x_index
    """
    assert sim.dim == 2
    Nx, Ny = int(sim.N_mesh[0]), int(sim.N_mesh[1])
    x, y = _grid_axes(sim.N_mesh, sim.boxsize)

    if y_index is None:
        y_index = Ny // 2
    if x_index is None:
        x_index = Nx // 2

    rho = sim.rho[t_index]
    E = sim.E_field[t_index]

    def get_img(key):
        if key == "rho":
            return rho
        if key == "Ex":
            return E[..., 0]
        if key == "Ey":
            return E[..., 1]
        if key == "E_mag":
            return jnp.sqrt(jnp.sum(E**2, axis=-1))
        raise ValueError(key)

    fig, axes = plt.subplots(len(what), 2, figsize=(10, 3 * len(what)), sharex="col")
    if len(what) == 1:
        axes = axes[None, :]

    for i, key in enumerate(what):
        img = get_img(key)
        # along x at fixed y
        axes[i, 0].plot(np.array(x), np.array(img[:, y_index]))
        axes[i, 0].set_title(f"{key}(x, y=y[{y_index}])")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel(key)

        # along y at fixed x
        axes[i, 1].plot(np.array(y), np.array(img[x_index, :]))
        axes[i, 1].set_title(f"{key}(x=x[{x_index}], y)")
        axes[i, 1].set_xlabel("y")
        axes[i, 1].set_ylabel(key)

    fig.suptitle(f"Lineouts @ t={float(sim.ts[t_index]):.3g}", y=1.02)
    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)


# -----------------------------
# B) Time series (invariants + residuals)
# -----------------------------
def plot_time_series_diagnostics(
    sim,
    eps=1e-12,
    save_path=None,
):
    """
    Plots:
      - Mass/charge from rho (integral over domain)
      - Momentum (from particles)
      - Kinetic energy, field energy, total energy
    """
    assert sim.positions is not None and sim.velocities is not None, "Need particle histories."
    assert sim.rho is not None and sim.E_field is not None, "Need rho and E_field histories."

    ts = sim.ts
    Nt = ts.shape[0]

    # Energies & momentum from histories
    K = jnp.zeros((Nt,), dtype=jnp.float64)
    U = jnp.zeros((Nt,), dtype=jnp.float64)
    Px = jnp.zeros((Nt,), dtype=jnp.float64)
    Py = jnp.zeros((Nt,), dtype=jnp.float64)
    M = jnp.zeros((Nt,), dtype=jnp.float64)

    for t in range(Nt):
        vel = sim.velocities[t]
        K = K.at[t].set(_kinetic_energy(sim.energy[t]))
        U = U.at[t].set(_field_energy(sim.E_field[t], sim.boxsize, sim.N_mesh, eps0=float(sim.eps0)))
        P = _momentum(sim.momentum[t])
        Px = Px.at[t].set(P[0])
        Py = Py.at[t].set(P[1])
        M = M.at[t].set(_mass_from_rho(sim.rho[t], sim.boxsize, sim.N_mesh))

    H = K + U

    # Relative drifts
    M_rel = (M - M[0]) / (jnp.abs(M[0]) + eps)
    H_rel = (H - H[0]) / (jnp.abs(H[0]) + eps)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ax = axes

    ax[0, 0].plot(np.array(ts), np.array(M))
    ax[0, 0].set_title("Total mass/charge (∫ rho)")
    ax[0, 0].set_ylabel("M")

    ax[0, 1].plot(np.array(ts), np.array(M_rel))
    ax[0, 1].set_title("Relative mass drift")
    ax[0, 1].set_ylabel("(M-M0)/|M0|")

    ax[1, 0].plot(np.array(ts), np.array(Px), label="Px")
    ax[1, 0].plot(np.array(ts), np.array(Py), label="Py")
    ax[1, 0].set_title("Total momentum (particles)")
    ax[1, 0].set_ylabel("P")
    ax[1, 0].legend()

    ax[1, 1].plot(np.array(ts), np.array(K), label="Kinetic")
    ax[1, 1].plot(np.array(ts), np.array(U), label="Field")
    ax[1, 1].plot(np.array(ts), np.array(H), label="Total")
    ax[1, 1].set_title("Energies")
    ax[1, 1].set_ylabel("Energy")
    ax[1, 1].legend()

    ax[2, 0].plot(np.array(ts), np.array(H_rel))
    ax[2, 0].set_title("Relative total energy drift")
    ax[2, 0].set_ylabel("(H-H0)/|H0|")
    ax[2, 0].set_xlabel("t")

    # simple field amplitude proxy
    E_rms = jnp.sqrt(jnp.mean(jnp.sum(sim.E_field.astype(jnp.float64) ** 2, axis=-1), axis=(1, 2)))
    ax[2, 1].plot(np.array(ts), np.array(E_rms))
    ax[2, 1].set_title("E RMS (grid)")
    ax[2, 1].set_ylabel("sqrt(mean(|E|^2))")
    ax[2, 1].set_xlabel("t")

    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)


# -----------------------------
# C) Animations for phase-space slices (2D projections)
# -----------------------------
def _select_particles_in_y_band(pos, y0, dy, Ly):
    """
    pos: (Np,2)
    select particles with y in [y0-dy/2, y0+dy/2] with periodic wrap
    returns mask (Np,)
    """
    y = pos[:, 1]
    # periodic distance on [0,Ly)
    dist = jnp.minimum(jnp.mod(y - y0, Ly), jnp.mod(y0 - y, Ly))
    return dist <= (dy / 2.0)


def animate_phase_slice_x_vx(
    sim,
    y0=None,
    dy=None,
    fps=30,
    k=1,
    vlim=None,
    s=0.4,
    alpha=0.4,
    max_points=20000,
    save_path=None,
    title_fmt="t = {t:.2f}",
):
    """
    Animate a *slice* of phase space by selecting particles in a y-band
    and plotting (x, v_x). This is the most useful 2D2V analogue of 1D1V (x,v).

    y0: center of y-band (default Ly/2)
    dy: band thickness (default ~2 cells)
    """
    assert sim.dim == 2
    ts = np.array(sim.ts)
    pos_hist = sim.positions
    vel_hist = sim.velocities
    Nt, Np, _ = pos_hist.shape
    Lx, Ly = float(sim.boxsize[0]), float(sim.boxsize[1])
    dx_y = float(sim.boxsize[1]) / int(sim.N_mesh[1])

    if y0 is None:
        y0 = 0.5 * Ly
    if dy is None:
        dy = 2.0 * dx_y

    if vlim is None:
        # crude but helpful default from initial thermal spread
        vmag = np.array(jnp.linalg.norm(vel_hist[0], axis=-1))
        vmax = float(np.percentile(vmag, 99))
        vlim = (-vmax, vmax)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(*vlim)
    ax.set_xlabel("x")
    ax.set_ylabel("v_x")

    sc = ax.scatter([], [], s=s, alpha=alpha)
    empty = np.empty((0, 2), dtype=float)

    def init():
        sc.set_offsets(empty)
        ax.set_title(title_fmt.format(t=ts[0]))
        return (sc,)

    def update(t):
        pos = pos_hist[t]
        vel = vel_hist[t]
        mask = _select_particles_in_y_band(pos, y0=y0, dy=dy, Ly=Ly)
        idx = np.array(jnp.where(mask)[0])

        if idx.size > max_points:
            idx = idx[:max_points]

        pts = np.column_stack([np.array(pos[idx, 0]), np.array(vel[idx, 0])]) if idx.size else empty
        sc.set_offsets(pts)
        ax.set_title(title_fmt.format(t=ts[t]) + f" | y-band: {y0:.2f}±{dy/2:.2f} (N={idx.size})")
        return (sc,)

    frames = range(0, Nt, k)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=True, interval=int(1000 / fps))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=-1))
        print(f"Saved animation to {save_path}")

    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_local_velocity_histogram(
    sim,
    x0=None,
    y0=None,
    rx=None,
    ry=None,
    bins=60,
    fps=30,
    k=1,
    vlim=None,
    save_path=None,
):
    """
    Animate a 2D histogram in velocity space (v_x, v_y) for particles
    inside a spatial probe box around (x0,y0).

    This is *very* informative for beams / heating / anisotropy.
    """
    assert sim.dim == 2
    ts = np.array(sim.ts)
    pos_hist = sim.positions
    vel_hist = sim.velocities
    Nt, Np, _ = pos_hist.shape
    Lx, Ly = float(sim.boxsize[0]), float(sim.boxsize[1])

    if x0 is None: x0 = 0.5 * Lx
    if y0 is None: y0 = 0.5 * Ly
    if rx is None: rx = float(sim.boxsize[0]) / int(sim.N_mesh[0]) * 2.0
    if ry is None: ry = float(sim.boxsize[1]) / int(sim.N_mesh[1]) * 2.0

    if vlim is None:
        vmag = np.array(jnp.linalg.norm(vel_hist[0], axis=-1))
        vmax = float(np.percentile(vmag, 99))
        vlim = (-vmax, vmax)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_y")

    # initialize empty image
    H0 = np.zeros((bins, bins))
    im = ax.imshow(H0, origin="lower", aspect="auto",
                   extent=[vlim[0], vlim[1], vlim[0], vlim[1]],
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax)

    def in_probe(pos):
        x = pos[:, 0]
        y = pos[:, 1]
        # periodic distance in each axis
        dxp = np.minimum(np.mod(x - x0, Lx), np.mod(x0 - x, Lx))
        dyp = np.minimum(np.mod(y - y0, Ly), np.mod(y0 - y, Ly))
        return (dxp <= rx) & (dyp <= ry)

    def update(t):
        pos = pos_hist[t]
        vel = vel_hist[t]
        mask = in_probe(pos)
        vv = np.array(vel[np.array(jnp.where(mask)[0])])

        if vv.size == 0:
            H = np.zeros((bins, bins))
        else:
            H, _, _ = np.histogram2d(vv[:, 0], vv[:, 1], bins=bins,
                                    range=[[vlim[0], vlim[1]], [vlim[0], vlim[1]]])

        im.set_data(H.T)  # histogram2d returns [xbin,ybin]; imshow expects rows=y
        im.set_clim(0, max(1.0, H.max()))
        ax.set_title(f"Local (v_x,v_y) @ t={ts[t]:.2f} | probe {(x0,y0)} ± {(rx,ry)} | N={vv.shape[0]}")
        return (im,)

    frames = range(0, Nt, k)
    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=int(1000 / fps))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=-1))
        print(f"Saved animation to {save_path}")

    plt.close(fig)
    return HTML(anim.to_jshtml())


# -----------------------------
# D) Frequency information (2D spectra)
# -----------------------------
def plot_2d_spectrum_snapshots(
    sim,
    field="rho",
    num=4,
    log=True,
    zero_mean=True,
    save_path=None,
):
    """
    Plot |FFT2(field)| at a few time indices.
    field: "rho" or "phi" (if you store it) or "E_mag"
    """
    assert sim.dim == 2
    ts = sim.ts
    Nt = ts.shape[0]
    Nx, Ny = int(sim.N_mesh[0]), int(sim.N_mesh[1])

    def get_field(t):
        if field == "rho":
            a = sim.rho[t]
        elif field == "E_mag":
            E = sim.E_field[t]
            a = jnp.sqrt(jnp.sum(E**2, axis=-1))
        else:
            raise ValueError(f"Unknown field: {field}")
        assert a.shape == (Nx, Ny)
        return a

    idxs = list(range(0, Nt, max(1, Nt // num)))
    if idxs[-1] != Nt - 1:
        idxs.append(Nt - 1)

    fig, axes = plt.subplots(1, len(idxs), figsize=(4.5 * len(idxs), 4), squeeze=False)
    axes = axes[0]

    for ax, t in zip(axes, idxs):
        a = get_field(t).astype(jnp.float64)
        if zero_mean:
            a = a - jnp.mean(a)

        A = jnp.fft.fft2(a)
        mag = jnp.abs(A)

        img = jnp.log(mag + 1e-12) if log else mag
        im = ax.imshow(np.array(jnp.fft.fftshift(img)), origin="lower", aspect="auto",
                       interpolation="nearest")
        ax.set_title(f"{field} |FFT2| @ t={float(ts[t]):.3g}")
        ax.set_xlabel("k_x (shifted)")
        ax.set_ylabel("k_y (shifted)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)


def plot_low_mode_growth_2d(
    sim,
    field="rho",
    modes=((1, 0), (0, 1), (1, 1), (2, 0), (0, 2)),
    zero_mean=True,
    semilogy=True,
    save_path=None,
):
    """
    Track a few low (kx,ky) Fourier coefficients over time.
    modes are integer index pairs in FFT grid coordinates:
      (mx,my) corresponds to coefficient A[mx,my] of fft2.
    """
    assert sim.dim == 2
    ts = sim.ts
    Nt = ts.shape[0]
    Nx, Ny = int(sim.N_mesh[0]), int(sim.N_mesh[1])

    def get_field(t):
        if field == "rho":
            a = sim.rho[t]
        elif field == "E_mag":
            E = sim.E_field[t]
            a = jnp.sqrt(jnp.sum(E**2, axis=-1))
        else:
            raise ValueError(field)
        return a.astype(jnp.float64)

    amps = []
    labels = []
    for (mx, my) in modes:
        labels.append(f"({mx},{my})")
        c = jnp.zeros((Nt,), dtype=jnp.float64)
        for t in range(Nt):
            a = get_field(t)
            if zero_mean:
                a = a - jnp.mean(a)
            A = jnp.fft.fft2(a)
            c = c.at[t].set(jnp.abs(A[mx % Nx, my % Ny]))
        amps.append(c)

    fig = plt.figure(figsize=(8, 4))
    for c, lab in zip(amps, labels):
        if semilogy:
            plt.semilogy(np.array(ts), np.array(c), label=lab)
        else:
            plt.plot(np.array(ts), np.array(c), label=lab)
    plt.xlabel("t")
    plt.ylabel(f"|FFT2({field})[mode]|")
    plt.title(f"Low-mode growth ({field})")
    plt.legend()
    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)

def _cell_area(boxsize, N_mesh):
    dx = float(boxsize[0]) / int(N_mesh[0])
    dy = float(boxsize[1]) / int(N_mesh[1])
    return dx * dy


def _compute_k_axes(boxsize, N_mesh):
    # physical k axes (cycles per length * 2pi), consistent with fft2 indices
    Lx, Ly = float(boxsize[0]), float(boxsize[1])
    Nx, Ny = int(N_mesh[0]), int(N_mesh[1])
    dx, dy = Lx / Nx, Ly / Ny
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    return kx, ky


def _fft2_power_zero_mean(rho_xy, zero_mean=True):
    a = np.asarray(rho_xy, dtype=np.float64)
    if zero_mean:
        a = a - a.mean()
    A = np.fft.fft2(a)
    P = (np.abs(A) ** 2)
    return P  # (Nx,Ny) power


def _exclude_dc(P):
    P2 = P.copy()
    P2[0, 0] = 0.0
    return P2


def _running_linear_fit(ts, y, tmin, tmax):
    """
    Fit y ~ a*t + b on the window [tmin,tmax]. Returns (a,b,r2).
    """
    t = np.asarray(ts)
    y = np.asarray(y)
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    X = np.vstack([t[mask], np.ones(mask.sum())]).T
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    a, b = coef[0], coef[1]
    yhat = X @ coef
    ss_res = np.sum((y[mask] - yhat) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


# ============================================================
# 1) "Classic" two-stream style: marginal phase-space movies
# ============================================================

def animate_marginal_phase_space(
    sim,
    which="x_vx",
    bins_x=220,
    bins_v=220,
    fps=30,
    k=1,
    vlim=None,
    log_scale=True,
    cmap="viridis",
    save_path=None,
    title_fmt="t = {t:.3g}",
):
    """
    Animate a coarse-grained marginal phase-space density using a 2D histogram.

    which:
      - "x_vx": f(x, v_x)   (classic two-stream look)
      - "y_vy": f(y, v_y)
      - "x_vy": f(x, v_y)
      - "y_vx": f(y, v_x)

    This integrates out the other two coordinates, so it often reveals vortices
    even when full 4D phase space is inaccessible.
    """
    assert sim.dim == 2
    ts = np.asarray(sim.ts)
    pos_hist = sim.positions
    vel_hist = sim.velocities
    Nt, Np, _ = pos_hist.shape

    Lx, Ly = float(sim.boxsize[0]), float(sim.boxsize[1])

    if which == "x_vx":
        L = Lx; pos_i = 0; vel_i = 0; xlabel = "x"; ylabel = "v_x"
    elif which == "y_vy":
        L = Ly; pos_i = 1; vel_i = 1; xlabel = "y"; ylabel = "v_y"
    elif which == "x_vy":
        L = Lx; pos_i = 0; vel_i = 1; xlabel = "x"; ylabel = "v_y"
    elif which == "y_vx":
        L = Ly; pos_i = 1; vel_i = 0; xlabel = "y"; ylabel = "v_x"
    else:
        raise ValueError(f"Unknown which={which}")

    if vlim is None:
        # robust default from first snapshot
        v0 = np.asarray(vel_hist[0, :, vel_i])
        vmax = float(np.percentile(np.abs(v0), 99.5))
        vlim = (-vmax, vmax)

    # histogram edges
    x_edges = np.linspace(0.0, L, bins_x + 1)
    v_edges = np.linspace(vlim[0], vlim[1], bins_v + 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # initial image
    H0 = np.zeros((bins_v, bins_x), dtype=float)  # imshow wants (rows=y, cols=x)
    im = ax.imshow(
        H0,
        origin="lower",
        aspect="auto",
        extent=[0.0, L, vlim[0], vlim[1]],
        interpolation="nearest",
        cmap=cmap,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("log(count+1)" if log_scale else "count")

    def update(t):
        x = np.asarray(pos_hist[t, :, pos_i])
        v = np.asarray(vel_hist[t, :, vel_i])

        # enforce periodic x in [0,L)
        x = np.mod(x, L)

        H, _, _ = np.histogram2d(v, x, bins=[v_edges, x_edges])  # (bins_v, bins_x)
        if log_scale:
            Z = np.log(H + 1.0)
        else:
            Z = H

        im.set_data(Z)
        im.set_clim(vmin=np.min(Z), vmax=np.max(Z) + 1e-12)
        ax.set_title(title_fmt.format(t=ts[t]) + f" | {which} | Np={Np}")
        return (im,)

    frames = range(0, Nt, k)
    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=int(1000 / fps))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=-1))
        print(f"Saved animation to {save_path}")

    plt.close(fig)
    return HTML(anim.to_jshtml())


# ============================================================
# 2) Instability detector: dominant mode in rho_k power
# ============================================================

def plot_max_mode_growth(
    sim,
    zero_mean=True,
    exclude_low_k_radius=0,  # set to 1 or 2 to ignore very lowest |k| if you want
    eps=1e-30,
    fit_window=None,         # e.g. (t_start, t_end) for linear fit in log-power
    save_path=None,
):
    """
    For each time:
      P(k) = |FFT2(rho - mean)|^2
      pick k* = argmax_{k != 0} P(k) (optionally excluding a disk around DC)
    Plots:
      - log(Pmax(t))
      - k* trajectory
      - and returns detected (mx*,my*) most frequently dominant.

    This is a robust "is it unstable?" detector because a true unstable mode
    yields a clean linear ramp in log(Pmax).
    """
    assert sim.rho is not None
    ts = np.asarray(sim.ts)
    rho_hist = sim.rho
    Nt, Nx, Ny = rho_hist.shape

    Pmax = np.zeros((Nt,), dtype=np.float64)
    kx_idx = np.zeros((Nt,), dtype=int)
    ky_idx = np.zeros((Nt,), dtype=int)

    for t in range(Nt):
        P = _fft2_power_zero_mean(rho_hist[t], zero_mean=zero_mean)
        P = _exclude_dc(P)

        if exclude_low_k_radius > 0:
            # exclude indices within a small L_infty ball around (0,0) AND around Nyquist wrap
            r = int(exclude_low_k_radius)
            P[:r+1, :r+1] = 0.0

        ind = np.unravel_index(np.argmax(P), P.shape)
        mx, my = int(ind[0]), int(ind[1])
        kx_idx[t] = mx
        ky_idx[t] = my
        Pmax[t] = float(P[mx, my] + eps)

    logP = np.log(Pmax)

    # "dominant" mode across time (most frequent)
    pairs = np.stack([kx_idx, ky_idx], axis=1)
    # find mode of pairs
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    dom_pair = uniq[np.argmax(counts)]
    dom_mx, dom_my = int(dom_pair[0]), int(dom_pair[1])

    # optional fit
    slope = intercept = r2 = np.nan
    if fit_window is not None:
        slope, intercept, r2 = _running_linear_fit(ts, logP, fit_window[0], fit_window[1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    ax = axes

    ax[0, 0].plot(ts, logP)
    ax[0, 0].set_title("log Pmax(t),  Pmax = max_{k≠0} |rho_k|^2")
    ax[0, 0].set_ylabel("log(Pmax)")

    if fit_window is not None and np.isfinite(slope):
        t0, t1 = fit_window
        ax[0, 0].axvspan(t0, t1, alpha=0.15)
        ax[0, 0].text(
            0.02, 0.05,
            f"fit window [{t0:.3g},{t1:.3g}]\nslope≈{slope:.3g} (so growth rate γ≈slope/2)\nR²≈{r2:.3g}",
            transform=ax[0, 0].transAxes,
            va="bottom",
        )

    ax[0, 1].plot(ts, Pmax)
    ax[0, 1].set_title("Pmax(t)")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel("Pmax")

    ax[1, 0].plot(ts, kx_idx, label="mx*")
    ax[1, 0].plot(ts, ky_idx, label="my*")
    ax[1, 0].set_title("argmax indices (mx*, my*) over time")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel("FFT index")
    ax[1, 0].legend()

    # show distribution of dominant pairs
    ax[1, 1].axis("off")
    ax[1, 1].text(
        0.0, 1.0,
        "Most frequent dominant mode:\n"
        f"(mx,my)=({dom_mx},{dom_my})\n\n"
        "Tip: track this specific mode's power too (see below).",
        va="top",
    )

    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)

    return (dom_mx, dom_my), (Pmax, (kx_idx, ky_idx), (slope, r2))


def plot_specific_mode_power(
    sim,
    mode=(1, 0),
    zero_mean=True,
    eps=1e-30,
    save_path=None,
):
    """
    Track P_k(t) = |rho_k(t)|^2 for a specific FFT index mode=(mx,my).
    This is the right "low-mode growth" observable (power, not amplitude).
    """
    ts = np.asarray(sim.ts)
    rho_hist = sim.rho
    Nt, Nx, Ny = rho_hist.shape
    mx, my = int(mode[0]) % Nx, int(mode[1]) % Ny

    Pk = np.zeros((Nt,), dtype=np.float64)
    for t in range(Nt):
        P = _fft2_power_zero_mean(rho_hist[t], zero_mean=zero_mean)
        Pk[t] = float(P[mx, my] + eps)

    fig = plt.figure(figsize=(8, 4))
    plt.semilogy(ts, Pk)
    plt.xlabel("t")
    plt.ylabel(r"$|\rho_k|^2$")
    plt.title(f"Mode power growth: (mx,my)=({mx},{my})")
    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)

    return Pk


# ============================================================
# 3) Field-energy growth + E_rms (best scalar instability signal)
# ============================================================

def plot_field_energy_growth(
    sim,
    save_path=None,
):
    """
    Plots:
      - U_E(t) = eps0/2 ∫|E|^2 dx dy
      - E_rms(t)
      - log(U_E(t)) if positive
    """
    assert sim.E_field is not None, "Need sim.E_field history for field-energy diagnostics."

    ts = np.asarray(sim.ts)
    E_hist = sim.E_field
    Nt = E_hist.shape[0]

    U = np.zeros((Nt,), dtype=np.float64)
    Erms = np.zeros((Nt,), dtype=np.float64)

    for t in range(Nt):
        E = np.asarray(E_hist[t])
        U[t] = _field_energy(E, sim.boxsize, sim.N_mesh, eps0=sim.eps0)
        Erms[t] = float(np.sqrt(np.mean(np.sum(E**2, axis=-1))))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(ts, U)
    axes[0].set_title("Field energy $U_E(t)$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("$U_E$")

    axes[1].plot(ts, Erms)
    axes[1].set_title("$E_{rms}(t)$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("$\\sqrt{\\langle |E|^2 \\rangle}$")

    # log plot (avoid -inf)
    Upos = np.where(U > 0, U, np.nan)
    axes[2].plot(ts, np.log(Upos))
    axes[2].set_title("log $U_E(t)$")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("log($U_E$)")

    plt.tight_layout()
    _maybe_save_or_show(fig, save_path=save_path)

    return U, Erms


# ============================================================
# 4) Energies in THREE separate plots (with extra info)
# ============================================================

def plot_energies_separate(
    sim,
    eps=1e-12,
    save_root=None,  # e.g. "out/energy" -> writes out/energy_kinetic.png etc.
):
    """
    Makes 3 separate figures:
      - kinetic energy K(t)
      - field energy U(t) (requires sim.E_field)
      - total energy H(t)=K+U (and relative drift)

    Also includes an additional panel for E_rms and relative drifts.
    """
    ts = np.asarray(sim.ts)
    vel_hist = sim.velocities
    assert vel_hist is not None, "Need sim.velocities history."
    Nt = vel_hist.shape[0]

    K = np.zeros((Nt,), dtype=np.float64)
    for t in range(Nt):
        K[t] = _kinetic_energy(sim.energy[t])

    U = None
    Erms = None
    if getattr(sim, "E_field", None) is not None:
        E_hist = sim.E_field
        U = np.zeros((Nt,), dtype=np.float64)
        Erms = np.zeros((Nt,), dtype=np.float64)
        for t in range(Nt):
            E = np.asarray(E_hist[t])
            U[t] = _field_energy(E, sim.boxsize, sim.N_mesh, eps0=sim.eps0)
            Erms[t] = float(np.sqrt(np.mean(np.sum(E**2, axis=-1))))
    else:
        # allow kinetic-only plotting
        U = np.zeros_like(K)
        Erms = np.full_like(K, np.nan)

    H = K + U
    K_rel = (K - K[0]) / (abs(K[0]) + eps)
    U_rel = (U - U[0]) / (abs(U[0]) + eps)
    H_rel = (H - H[0]) / (abs(H[0]) + eps)

    # ---- Kinetic plot
    figK = plt.figure(figsize=(8, 4))
    plt.plot(ts, K)
    plt.xlabel("t")
    plt.ylabel("K")
    plt.title("Kinetic energy K(t)")
    plt.tight_layout()
    _maybe_save_or_show(figK, None if save_root is None else f"{save_root}_kinetic.png")

    # ---- Field plot
    figU, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(ts, U)
    ax[0].set_title("Field energy U(t)")
    ax[0].set_ylabel("U")

    ax[1].plot(ts, Erms)
    ax[1].set_title("E_rms(t)")
    ax[1].set_ylabel("E_rms")
    ax[1].set_xlabel("t")

    plt.tight_layout()
    _maybe_save_or_show(figU, None if save_root is None else f"{save_root}_field.png")

    # ---- Total plot (+ drift info)
    figH, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(ts, H, label="H=K+U")
    ax[0].plot(ts, K, label="K", alpha=0.8)
    ax[0].plot(ts, U, label="U", alpha=0.8)
    ax[0].set_title("Total energy and components")
    ax[0].set_ylabel("Energy")
    ax[0].legend()

    ax[1].plot(ts, H_rel, label="(H-H0)/|H0|")
    ax[1].plot(ts, K_rel, label="(K-K0)/|K0|", alpha=0.8)
    ax[1].plot(ts, U_rel, label="(U-U0)/|U0|", alpha=0.8)
    ax[1].set_title("Relative drifts")
    ax[1].set_ylabel("relative drift")
    ax[1].set_xlabel("t")
    ax[1].legend()

    plt.tight_layout()
    _maybe_save_or_show(figH, None if save_root is None else f"{save_root}_total.png")

    return K, U, H