import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from plotting import plot_2d_spectrum_snapshots, plot_fields_snapshot, plot_lineouts, plot_low_mode_growth_2d, plot_time_series_diagnostics, animate_phase_slice_x_vx, animate_local_velocity_histogram
import matplotlib.pyplot as plt

# Simulation parameters
N_particles = 100000  # Number of particles
N_mesh = (400,400)  # Number of mesh cells
t1 = 30  # time at which simulation ends
dt = 0.1  # timestep
boxsize = jnp.array([10*jnp.pi,10*jnp.pi])  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
dim = 2

pic = PICSimulation(dim, boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

key = jax.random.key(42)
y0 = pic.create_y0(key)

pic = pic.run_simulation(y0)

plot_time_series_diagnostics(pic, save_path="plots/zir/diag_timeseries.png")
plot_fields_snapshot(pic, t_index=-1, save_path="plots/zir/fields_final.png")
plot_lineouts(pic, t_index=-1, save_path="plots/zir/lineouts_final.png")

plot_2d_spectrum_snapshots(pic, field="rho", save_path="plots/zir/rho_fft2.png")
plot_low_mode_growth_2d(pic, field="rho", save_path="plots/zir/rho_lowmodes.png")

# Animations (works in notebooks)
animate_phase_slice_x_vx(pic, save_path="plots/zir/x_vx_slice.mp4")
animate_local_velocity_histogram(pic, save_path="plots/zir/local_v_hist.mp4")