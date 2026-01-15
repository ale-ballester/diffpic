import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import importlib

from jax.nn.initializers import glorot_uniform

def make_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def create_external_field(ts,A,phi_t,phi_x,n,m,boxsize,N_mesh):
    omega = 2 * jnp.pi * n
    k = 2 * jnp.pi * m / boxsize
    space_grid = jnp.linspace(0,boxsize,N_mesh,endpoint=False)
    u = A * jnp.sin(ts[:, None] * omega + phi_t) * jnp.sin(space_grid * k + phi_x)
    return u