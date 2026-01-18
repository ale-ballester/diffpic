import jax
import jax.numpy as jnp

def loss_metric(pic):
    energy = jnp.mean((pic.E_field)**2)
    return energy

def phase_invariant_loss(pic, *,
                         K_rho=10,
                         lam_u=1e-3,
                         lam_dc=0.0):
    """
    pic: output of run_simulation with attributes:
         - pic.rho: (Nt, N_mesh) or (Nt, N_mesh, 1)
         - pic.ts:  (Nt,)
    model: your FourierActuator (open-loop), callable on integer step index n -> E_ext(x)

    Returns scalar loss.
    """

    rho = pic.rho
    if rho.ndim == 3:
        rho = rho[..., 0]  # (Nt, N_mesh)

    # Low-k density power (phase-invariant)
    rho_k = jnp.fft.rfft(rho, axis=-1)  # (Nt, N_mesh//2+1) complex
    rho_k = rho_k.at[:, 0].set(0.0)     # remove DC
    K_rho = int(min(K_rho, rho_k.shape[-1] - 1))
    rho_power = jnp.mean(jnp.abs(rho_k[:, 1:K_rho+1])**2)

    # Control energy penalty
    Nt = pic.ts.shape[0]
    Eext = pic.E_ext          # expects model(n) -> (N_mesh,) or (N_mesh,1)
    if Eext.ndim == 3:
        Eext = Eext[..., 0]
    u_energy = jnp.mean(Eext**2)

    # DC penalty on control (discourage net acceleration / bias)
    u_dc = jnp.mean(Eext, axis=-1)          # mean over space -> (Nt,)
    dc_pen = jnp.mean(u_dc**2)

    return rho_power + lam_u * u_energy + lam_dc * dc_pen