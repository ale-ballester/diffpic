import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from control import DissipativeModeFeedbackActuator
from losses import loss_metric, loss_metric_density_modes, loss_metric_stable
from optimize import Optimizer
from plotting import scatter_animation, plot_pde_solution, plot_modes
import matplotlib.pyplot as plt

# Simulation parameters
N_particles = 40000  # Number of particles
Nh = int(N_particles / 2)
N_mesh = 256  # Number of mesh cells
t1 = 20  # time at which simulation ends
dt = 0.1  # timestep
boxsize = 10*jnp.pi  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 2.4  # beam velocity
vth = 0.5  # beam width
pos_sample = False
K = 4
seed_ic = 10
n_modes_space_in = 10
n_modes_space_out = 10

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

key = jax.random.PRNGKey(9)

E_control = DissipativeModeFeedbackActuator(
    N_mesh=pic.N_mesh,
    boxsize=pic.boxsize,
    n_modes_space_in=n_modes_space_in,
    n_modes_space_out=n_modes_space_out,
    width=32,
    depth=2,
    u_max=None,
    include_dc=False,
    closed_loop=True,
    key=key
)

key = jax.random.key(seed_ic)
y0 = pic.create_y0(key)

optimizer = Optimizer(pic=pic,model=E_control,K=K,y0=y0,loss_metric=loss_metric_stable,lr=1e-2,save_dir="model_cl_dis/")

E_control, train_losses, _ = optimizer.train(
    n_steps=200, 
    save_every=100, 
    seed=0, 
    print_status=True)

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, vb, vth, dt, 2*t1, t0=0, higher_moments=True)

E_control = DissipativeModeFeedbackActuator.load_model("model_cl_dis/model_checkpoint_final")

key = jax.random.key(1024)
y0 = pic.create_y0(key)

pic = pic.run_simulation(y0,E_control=E_control)

E_control_vmappable = lambda n,x: E_control(n,state=x)
u = jax.vmap(E_control_vmappable,in_axes=(0,0))(jnp.arange(pic.ts.shape[0]),jnp.fft.rfft(pic.momentum))

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/trained_cl_dis/scatter.mp4")

plot_pde_solution(pic.ts, u, boxsize, name=r"External field", label=r"$E_{ext}$", save_path="plots/trained_cl_dis/external_field.png")
plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/trained_cl_dis/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/trained_cl_dis/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/trained_cl_dis/energy.png")

plot_modes(pic.ts, u, max_mode_spect=n_modes_space_out, max_mode_time=n_modes_space_out, boxsize=boxsize, name=r"External field", label=r"$\hat E_{ext}$", num=6, zero_mean=True, save_path="plots/trained_cl_dis/external_field_modes.png")
plot_modes(pic.ts, pic.rho, max_mode_spect=100, max_mode_time=5, boxsize=boxsize, name=r"Density", label=r"$\hat\rho_k$", num=6, zero_mean=True, save_path="plots/trained_cl_dis/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=100, max_mode_time=5, boxsize=boxsize, name=r"Momentum", label=r"$\hat\mathcal{{p}}_k$", num=6, zero_mean=True, save_path="plots/trained_cl_dis/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=100, max_mode_time=5, boxsize=boxsize, name=r"Energy", label=r"$\hat\mathcal{{E}}_k$", num=6, zero_mean=True, save_path="plots/trained_cl_dis/energy_modes.png")

plt.figure()
plt.plot(pic.ts,jnp.sum(pic.energy,axis=-1))
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy Evolution")
plt.tight_layout()
plt.savefig("plots/trained_cl_dis/energy_evolution.png", dpi=300)

plt.figure()
plt.semilogy(train_losses)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Training loss")
plt.tight_layout()

plt.savefig("plots/trained_cl_dis/train_losses.png", dpi=300)