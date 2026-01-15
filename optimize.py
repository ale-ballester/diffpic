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

class Optimizer():
    def __init__(self, pic,
                       y0,
                       model,
                       loss_metric,
                       loss_kwargs=None,
                       lr=1e-4,
                       optim=None,
                       save_dir="model/", 
                       save_name="model_checkpoint",
                       seed=0):
        self.pic = pic
        self.y0 = y0
        self.model = model
        if loss_kwargs is None: loss_kwargs = {}
        self.loss_metric = loss_metric
        self.loss = lambda model: self.loss_function(model, **loss_kwargs) # Same signature as L2_loss, but different implementation
        self.grad_loss = eqx.filter_value_and_grad(self.loss) # Do NOT mutate loss after this point, it is jitted already
        self.lr = lr
        if optim is None:
            self.optim = optax.adam(lr)
        else:
            self.optim = optim(lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name

    def loss_function(self, model, **kwargs):
        pic = self.pic.run_simulation(self.y0,E_control=model)
        loss = self.loss_metric(pic, **kwargs)
        return loss
    
    def make_step(self, model, opt_state):
        loss, grads = self.grad_loss(model)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train(self, n_steps, save_every=100, seed=0, print_status=True):
        make_step = eqx.filter_jit(self.make_step) # Do NOT mutate anything inside self.make_step from this point on, it is jitted already    

        make_dir(self.save_dir)

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        train_losses = []
        valid_losses = []

        loader_key = jax.random.PRNGKey(seed)

        for step in range(n_steps):            
            if print_status:
                print("--------------------")
                print(f"Step: {step}")
            loader_key, train_loader_key = jax.random.split(loader_key)
            start = time.time()
            loss, self.model, opt_state = make_step(self.model, opt_state)
            end = time.time()
            train_losses.append(loss)
            loader_key, valid_loader_key = jax.random.split(loader_key)
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