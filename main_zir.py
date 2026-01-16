import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from plotting import scatter_animation, plot_pde_solution, plot_modes

# Simulation parameters
N_particles = 40000  # Number of particles
N_mesh = 400  # Number of mesh cells
t1 = 30  # time at which simulation ends
dt = 0.1  # timestep
boxsize = 10*jnp.pi  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
pos_sample = False


key = jax.random.key(0)
key1, key2, key3 = jax.random.split(key, num=3)

if pos_sample:
    pos = jax.random.uniform(key1, (N_particles, 1)) * boxsize
else:
    pos = jnp.zeros(N_particles)
    for i in range(N_particles):
        pos = pos.at[i].set(i*boxsize/N_particles)
    pos = jax.random.choice(key1, pos, shape=pos.shape, replace=False)
    pos = pos[:,None]

vel = vth * jax.random.normal(key2, (N_particles, 1)) + vb
Nh = int(N_particles / 2)
vel = vel.at[Nh:].set(-1*vel[Nh:])
vel = vel - jnp.mean(vel)

y0 = (pos, vel)

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, higher_moments=True)

pic = pic.run_simulation(y0)

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/zir/scatter.mp4")

plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/zir/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/zir/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/zir/energy.png")

plot_modes(pic.ts, pic.rho, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Density", label=r"$\hat\rho_k$", num=4, zero_mean=True, save_path="plots/zir/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Momentum", label=r"$\hat\mathcal{{p}}_k$", num=4, zero_mean=True, save_path="plots/zir/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Energy", label=r"$\hat\mathcal{{E}}_k$", num=4, zero_mean=True, save_path="plots/zir/energy_modes.png")