import time
import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dataloader import DataLoader
from utils import make_dir

def grad_diagnostics(grads):
    # Collect gradient leaves (JAX-safe)
    leaves = jax.tree_util.tree_leaves(grads)

    # Filter out None leaves
    leaves = [g for g in leaves if g is not None]

    # Number of gradient arrays
    num_leaves = len(leaves)

    # Flatten all gradients into one vector
    flat_grads = jnp.concatenate([jnp.ravel(g) for g in leaves])

    # Compute stats
    max_abs = jnp.max(jnp.abs(flat_grads))
    mean_abs = jnp.mean(jnp.abs(flat_grads))
    l2_norm = jnp.linalg.norm(flat_grads)

    # JAX-safe printing
    jax.debug.print(
        "Grad diagnostics | leaves: {n}, max|g|: {mx:.3e}, mean|g|: {mn:.3e}, ||g||₂: {l2:.3e}",
        n=num_leaves,
        mx=max_abs,
        mn=mean_abs,
        l2=l2_norm,
    )

class Optimizer():
    def __init__(self, pic,
                       model,
                       loss_metric,
                       loss_kwargs=None,
                       K=None,
                       y0=None,
                       lr=1e-4,
                       optim=None,
                       save_dir="model/", 
                       save_name="model_checkpoint",
                       seed=0):
        self.pic = pic
        self.K = K
        self.y0 = y0
        if self.y0 is not None:
            if self.y0[0].ndim != 2:
                raise ValueError(f"Expected unbatched y0 (N,1). Got {self.y0[0].shape}")
        self.model = model
        if loss_kwargs is None: loss_kwargs = {}
        self.loss_metric = loss_metric
        self.loss = lambda model, y0: self.loss_function(model, y0, **loss_kwargs) # Same signature as L2_loss, but different implementation
        self.grad_loss = eqx.filter_value_and_grad(self.loss) # Do NOT mutate loss after this point, it is jitted already
        self.lr = lr
        if optim is None:
            self.optim = optax.adam(lr)
        else:
            self.optim = optim(lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name
    
    def loss_function(self, model, y0, **kwargs):
        pos, vel = y0
        if pos.ndim == 2:   # (N,1) single IC
            pic = self.pic.run_simulation((pos, vel), E_control=model)
            return self.loss_metric(pic, **kwargs)

        # batched: (K,N,1)
        run_one = lambda pos_i, vel_i: self.pic.run_simulation((pos_i, vel_i), E_control=model)
        pic_batch = jax.vmap(run_one)(pos, vel)
        losses = jax.vmap(lambda pic: self.loss_metric(pic, **kwargs))(pic_batch)
        return losses.mean()
    
    def make_step(self, model, opt_state, y0):
        loss, grads = self.grad_loss(model, y0)
        #grad_diagnostics(grads)
        #jax.debug.print("{grads}",grads=grads)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train(self, n_steps, save_every=100, seed=0, print_status=True):
        make_step = eqx.filter_jit(self.make_step) # Do NOT mutate anything inside self.make_step from this point on, it is jitted already    

        make_dir(self.save_dir)

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        train_losses = []
        valid_losses = []

        ic_key = jax.random.PRNGKey(seed)

        for step in range(n_steps):            
            if print_status:
                print("--------------------")
                print(f"Step: {step}")            
            start = time.time()
            if self.K is not None:
                ic_key, subkey = jax.random.split(ic_key)
                ic_key_arr = jax.random.split(subkey, self.K)
                y0 = jax.vmap(self.pic.create_y0)(ic_key_arr)
            else:
                y0 = self.y0
            loss, self.model, opt_state = make_step(self.model, opt_state, y0)
            end = time.time()
            train_losses.append(loss)
            if print_status: print(f"Train loss: {loss}")
            if step % save_every == 0 and step > 0 and step < n_steps-1:
                if print_status: print(f"Saving model at step {step}")
                checkpoint_name = self.save_dir+self.save_name+f"_{step}"
                self.model.save_model(checkpoint_name)
        if print_status: print("Training complete.")
        checkpoint_name = self.save_dir+self.save_name+"_final"
        if print_status: print(f"Saving model at {checkpoint_name}")
        self.model.save_model(checkpoint_name)

        return self.model, train_losses, valid_losses