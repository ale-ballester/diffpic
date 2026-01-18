import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from control import ModeFeedbackActuator
from losses import loss_metric, phase_invariant_loss
from optimize import Optimizer
from plotting import scatter_animation, plot_pde_solution, plot_modes
import matplotlib.pyplot as plt

# Simulation parameters
N_particles = 100000  # Number of particles
Nh = int(N_particles / 2)
N_mesh = 400  # Number of mesh cells
t1 = 30  # time at which simulation ends
dt = 0.1  # timestep
boxsize = 10*jnp.pi  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
pos_sample = False

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

E_control = ModeFeedbackActuator.load_model("model_cl/model_checkpoint_final")

key = jax.random.key(seed_ic)
y0 = pic.create_y0(key)

pic = pic.run_simulation(y0,E_control=E_control)

#u = jax.vmap(jax.vmap(E_control,in_axes=(0,None)),in_axes=(None,0))(jnp.arange(pic.ts.shape[0]),jnp.fft.rfft(pic.rho))

#plot_pde_solution(pic.ts, u, boxsize, name=r"External field", label=r"$E_{ext}$", save_path="plots/trained_cl/external_field.png")

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/trained_cl/scatter.mp4")

plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/trained_cl/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/trained_cl/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/trained_cl/energy.png")

plot_modes(pic.ts, pic.rho, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Density", label=r"$\hat\rho_k$", num=4, zero_mean=True, save_path="plots/trained_cl/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Momentum", label=r"$\hat\mathcal{{p}}_k$", num=4, zero_mean=True, save_path="plots/trained_cl/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Energy", label=r"$\hat\mathcal{{E}}_k$", num=4, zero_mean=True, save_path="plots/trained_cl/energy_modes.png")

plt.figure()
plt.plot(train_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Training loss")
plt.tight_layout()

plt.savefig("plots/trained_cl/train_losses.png", dpi=300)