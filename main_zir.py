import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from plotting import plot_2d_spectrum_snapshots, plot_fields_snapshot, plot_lineouts, plot_low_mode_growth_2d, plot_time_series_diagnostics, animate_phase_slice_x_vx, animate_local_velocity_histogram, plot_energies_separate, plot_field_energy_growth, plot_max_mode_growth, plot_specific_mode_power, animate_marginal_phase_space
import matplotlib.pyplot as plt
from utils import timer, _block_until_ready_pytree

jax.config.update('jax_enable_x64', True)

print(jax.devices())

# Simulation parameters
N_particles = 51200  # Number of particles
N_mesh = (32,32)  # Number of mesh cells
t1 = 30  # time at which simulation ends
dt = 0.1  # timestep
boxsize = jnp.array([3*jnp.pi,3*jnp.pi])  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
dim = 2

pic = PICSimulation(dim, boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

key = jax.random.key(42)
y0 = pic.create_y0(key)

@jax.jit
def give_rho(pic, y0):
    return pic.run_simulation(y0).rho

timed_sim = timer(lambda y0: give_rho(pic, y0))


rho0, t_run = timed_sim(y0)
_block_until_ready_pytree(rho0)   

times = []
for i in range(5):
    rho, t = timed_sim(y0)
    _block_until_ready_pytree(rho)
    times.append(t)
    print(f"run {i}: {t:.6f} s")
    print(f"We have a test rho value: {rho[0,0,0]}")

print(f"mean run time: {sum(times)/len(times):.6f} s")

pic = pic.run_simulation(y0)

plot_time_series_diagnostics(pic, save_path="plots/zir/diag_timeseries.png")
plot_fields_snapshot(pic, t_index=-1, save_path="plots/zir/fields_final.png")
plot_lineouts(pic, t_index=-1, save_path="plots/zir/lineouts_final.png")

plot_2d_spectrum_snapshots(pic, field="rho", save_path="plots/zir/rho_fft2.png")
plot_low_mode_growth_2d(pic, field="rho", save_path="plots/zir/rho_lowmodes.png")
plot_field_energy_growth(pic, save_path="plots/zir/field_energy_growth.png")
plot_energies_separate(pic, save_root="plots/zir/energy")

(dom_mx, dom_my), info = plot_max_mode_growth(
    pic,
    fit_window=(5.0, 15.0),     # pick a window where you expect linear-stage growth
    save_path="plots/zir/rho_maxmode_growth.png"
)

plot_specific_mode_power(pic, mode=(dom_mx, dom_my), save_path="plots/zir/rho_dom_mode_power.png")

# Animations (works in notebooks)
animate_phase_slice_x_vx(pic, save_path="plots/zir/x_vx_slice.mp4")
animate_local_velocity_histogram(pic, save_path="plots/zir/local_v_hist.mp4")

animate_marginal_phase_space(pic, which="x_vx", save_path="plots/zir/fxvx.mp4", k=2, fps=30)