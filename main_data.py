import jax
import jax.numpy as jnp
from pic_simulation import PICSimulation
from plotting import scatter_animation, plot_pde_solution, plot_modes
from utils import create_external_field, make_dir
from rom import fom_state_to_rom_state
import pickle
import matplotlib.pyplot as plt

# Simulation parameters
N_particles = 40000  # Number of particles
N_mesh = 400  # Number of mesh cells
t1 = 5  # time at which simulation ends
dt = 1e-3  # timestep
boxsize = 50  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 3  # beam velocity
vth = 1  # beam width
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

y0 = (pos, vel)

def generate_mode_table(
    n_modes,
    n_max,
    m_max,
    amp_scale=1.0,
    key=jax.random.PRNGKey(0),
):
    """
    Returns array of shape (n_modes, 4):
    [n, m, amplitude, phase]
    """

    key_n, key_m, key_a, key_pt, key_px = jax.random.split(key, 5)

    # Integer mode indices
    n = jax.random.randint(key_n, (n_modes,), 1, n_max + 1) # time
    m = jax.random.randint(key_m, (n_modes,), 1, n_max + 1) # space

    # Amplitudes and phases
    amp = amp_scale * jax.random.uniform(key_a, (n_modes,))
    phi_t = 2 * jnp.pi * jax.random.uniform(key_pt, (n_modes,))
    phi_x = 2 * jnp.pi * jax.random.uniform(key_px, (n_modes,))

    modes = jnp.stack([n, m, amp, phi_t, phi_x], axis=1)
    return modes

K = 3
modes = generate_mode_table(K, K, K)
print("Modes shape: ", modes.shape)
print("Modes:\n", modes)

def field_fft(mode):
    n,m,A,phi_t,phi_x = mode
    pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, E_control=None, higher_moments=True)
    u = create_external_field(ts=pic.ts,A=A,phi_t=phi_t,phi_x=phi_x,n=n,m=m,boxsize=pic.boxsize,N_mesh=pic.N_mesh)
    u_k = jnp.fft.rfft(u,axis=-1)[:,1:K+1]
    x = jnp.concatenate([jnp.real(u_k), jnp.imag(u_k)], axis=-1)
    return x

u_rom_data = jax.vmap(field_fft)(modes)
print("u_rom_data.shape: ", u_rom_data.shape)

fom_state_to_rom_state_vmapped = jax.vmap(fom_state_to_rom_state, in_axes=(0,0,0,None))

def run_once(u):
    pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, E_control=None, higher_moments=True)
    pic = pic.run_simulation(y0, u)
    rho = pic.rho
    mom = pic.momentum
    ene = pic.energy
    y_rom = fom_state_to_rom_state_vmapped(rho, mom, ene, K)
    return y_rom

y_rom_data = jax.vmap(run_once)(u_rom_data)        

print("y_rom_data.shape: ", y_rom_data.shape)

make_dir("data")
with open("data/dataset.pkl", "wb") as f:
    pickle.dump(y_rom_data, f)

plt.plot(u_rom_data[-1])
plt.show()

with open("data/modes.pkl", "wb") as f:
    pickle.dump(u_rom_data, f)

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, dt, t1, t0=0, E_control=None, higher_moments=True)

with open("data/params.pkl", "wb") as f:
    pickle.dump(pic, f)