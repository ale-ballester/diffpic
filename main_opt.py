import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from control import FourierActuator
from optimize import Optimizer
from plotting import scatter_animation, plot_pde_solution, plot_modes
import matplotlib.pyplot as plt

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
n_modes_time = 1
n_modes_space = 4


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

y0 = (pos, vel)

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, higher_moments=True)

key = jax.random.PRNGKey(0)

E_control = FourierActuator(
    Nt=pic.n_steps,
    N_mesh=pic.N_mesh,
    boxsize=pic.boxsize,
    n_modes_time=n_modes_time,      # static field (time-DC only)
    n_modes_space=n_modes_space,     # m = 0, 1  (DC + first harmonic)
    key=key,             # random small init
    init_scale=1e-4,     # small initial field
    zero=False,
    closed_loop=False,
)

def loss_metric(pic):
    energy = jnp.mean((pic.E_field)**2)
    return energy

optimizer = Optimizer(pic=pic,y0=y0,model=E_control,loss_metric=loss_metric,lr=1e-1)

E_control, train_losses, _ = optimizer.train(
    n_steps=200, 
    save_every=100, 
    seed=0, 
    print_status=True)

plt.figure()
plt.plot(train_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Training loss")
plt.tight_layout()

plt.savefig("plots/trained/train_losses.png", dpi=300)

print("Control modes:\n", E_control.get_modes_summary())

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, higher_moments=True)

pic = pic.run_simulation(y0,E_control=E_control)

u = jax.vmap(E_control)(pic.ts)

plot_pde_solution(pic.ts, u, boxsize, name=r"External field", label=r"$E_{ext}$", save_path="plots/trained/external_field.png")

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/trained/scatter.mp4")

plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/trained/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/trained/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/trained/energy.png")

plot_modes(pic.ts, pic.rho, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Density", label=r"$\hat\rho_k$", num=4, zero_mean=True, save_path="plots/trained/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Momentum", label=r"$\hat\mathcal{{p}}_k$", num=4, zero_mean=True, save_path="plots/trained/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, name=r"Energy", label=r"$\hat\mathcal{{E}}_k$", num=4, zero_mean=True, save_path="plots/trained/energy_modes.png")