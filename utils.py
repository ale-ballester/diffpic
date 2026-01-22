import os
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import equinox as eqx
import importlib
import functools
import time

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

def _block_until_ready_pytree(x):
    # Block on all JAX arrays inside x (works for pytrees)
    leaves = jtu.tree_leaves(x)
    # If there are no leaves, nothing to block on
    if not leaves:
        return
    # Block on any leaf that has block_until_ready; typically all jax.Arrays do
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return
    # If leaves exist but none are JAX arrays, nothing to block on
    return

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        _block_until_ready_pytree(out)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"Execution took {dt:.6f}s")
        return out, dt
    return wrapper