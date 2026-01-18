import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from control import FourierActuator
from plotting import scatter_animation, plot_pde_solution, plot_modes
import matplotlib.pyplot as plt

# Simulation parameters
N_particles = 100000  # Number of particles
Nh = int(N_particles / 2)
N_mesh = 4000  # Number of mesh cells
t1 = 30  # time at which simulation ends
dt = 0.1  # timestep
boxsize = 10*jnp.pi  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
pos_sample = False
n_modes_time = 1
n_modes_space = 4

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

key = jax.random.key(42)
y0 = pic.create_y0(key)

E_control = FourierActuator.load_model("model/model_checkpoint_final")

print(E_control)

print("Control modes:\n", E_control.get_modes_summary())

pic = pic.run_simulation(y0,E_control=E_control)

u = jax.vmap(E_control)(jnp.arange(pic.ts.shape[0]))

plot_pde_solution(pic.ts, u, boxsize, name=r"External field", label=r"$E_{ext}$", save_path="plots/test/external_field.png")

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/test/scatter.mp4")

plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/test/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/test/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/test/energy.png")

plot_modes(pic.ts, pic.rho, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Density", label=r"$\hat\rho_k$", num=4, zero_mean=True, save_path="plots/test/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Momentum", label=r"$\hat\mathcal{{p}}_k$", num=4, zero_mean=True, save_path="plots/test/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Energy", label=r"$\hat\mathcal{{E}}_k$", num=4, zero_mean=True, save_path="plots/test/energy_modes.png")