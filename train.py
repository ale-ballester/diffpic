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

class Trainer():
    def __init__(self, model=None,
                       loss="L2",
                       loss_kwargs=None,
                       lr=1e-4,
                       optim=None,
                       save_dir="model/", 
                       save_name="model_checkpoint",
                       seed=0):
        self.model = model
        if loss_kwargs is None: loss_kwargs = {}
        if loss == "L2":
            self.loss = lambda model, ti, yi, ui: self.L2_loss(model, ti, yi, ui, **loss_kwargs)
        else:
            self.loss = lambda model, ti, yi, ui: loss(model, ti, yi, ui, **loss_kwargs) # Same signature as L2_loss, but different implementation
        self.grad_loss = eqx.filter_value_and_grad(self.loss) # Do NOT mutate loss after this point, it is jitted already
        self.lr = lr
        if optim is None:
            self.optim = optax.adam(lr)
        else:
            self.optim = optim(lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name
    
    def L2_loss(self, model, ti, yi, ui, **kwargs):
        """
        Computes the L2 loss between the model predictions and the true values.
        Args:
            model: The model to evaluate.
            ti: The time points at which to evaluate the model.
            yi: The true values at the time points, with shape (batch_size, time_steps, dim).
        Returns:
            The L2 loss value.
        """
        y_pred = jax.vmap(model, in_axes=(None, 0, 0))(ti, yi[:,0,:], ui)
        loss = ((yi - y_pred) ** 2).mean(axis=(1,2)).mean()
        return loss
    
    def make_step(self, ti, yi, ui, model, opt_state):
        loss, grads = self.grad_loss(model, ti, yi, ui)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    def create_dataloader(self, ts, data_in, data_out, permute=True, seed=0):
        dataloader = DataLoader(ts, data_in, data_out, permute=permute)
        return dataloader

    def train(self, ts, data_in, data_out, n_epochs, bs, time_windows=None, n0=None, nrand=None, save_every=100, seed=0, print_status=True, save_plots=False, permute=True):
        make_step = eqx.filter_jit(self.make_step) # Do NOT mutate anything inside self.make_step from this point on, it is jitted already    

        N = data_in.shape[0]

        dl_train = self.create_dataloader(ts, data_in, data_out, permute=permute, seed=seed)

        make_dir(self.save_dir)
        make_dir(self.save_dir + "png")

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        steps_per_epoch = max([int(jnp.floor(N / bs)),1]) # Could also be ceil, but then last batch can roll over
        #steps_valid = max([int(jnp.floor(N_valid / bs_valid)),1]) # Could also be ceil, but then last batch can roll over

        N_time_schedules = len(time_windows) if time_windows is not None else 1
        schedule = [(None, None)]
        if time_windows is not None:
            if nrand is None:
                nrand = [None] * N_time_schedules
            elif len(nrand) != N_time_schedules:
                raise ValueError("nrand must have the same length as time_windows.")
            schedule = list(zip(time_windows, nrand))

        train_losses = []
        valid_losses = []

        loader_key = jax.random.PRNGKey(seed)

        for epoch in range(n_epochs):
            index = epoch // (n_epochs // N_time_schedules)
            if epoch % (n_epochs // N_time_schedules) == 0 and index < N_time_schedules:
                t_now, nrand_now = schedule[index]
                n_t = round(jnp.floor((t_now - ts[0]) / (ts[-1] - ts[0]) * (len(ts))))
                if print_status:
                    print("--------------------\n")
                    print(f"Training for time window {ts[n_t]} ({n_t} total samples), with {nrand_now} samples.\n")
            if print_status:
                print("--------------------")
                print(f"Epoch: {epoch}")
            train_loss_epoch = 0
            valid_loss_epoch = 0
            loader_key, train_loader_key = jax.random.split(loader_key)
            for step, batch in zip(range(steps_per_epoch),dl_train(bs, key=train_loader_key,n0=n0,n1=n_t,nrand=nrand_now)):
                start = time.time()
                ts, ui, yi = batch
                loss, self.model, opt_state = make_step(ts, yi, ui, self.model, opt_state)
                train_loss_epoch += loss
                end = time.time()
            train_loss_epoch /= steps_per_epoch
            train_losses.append(train_loss_epoch)
            loader_key, valid_loader_key = jax.random.split(loader_key)
            #for step, yi in zip(range(steps_valid),dl_valid(bs, key=train_loader_key)):
                ### TODO: Implement validation loss function
                #loss = val_loss(model, ts_valid, yi)
            #    valid_loss_epoch += loss
            #valid_loss_epoch /= steps_valid
            #valid_losses.append(valid_loss_epoch)
            if print_status: print(f"Train loss: {train_loss_epoch}, Valid loss: {valid_loss_epoch}")
            if epoch % save_every == 0 and epoch > 0 and epoch < n_epochs-1:
                if print_status: print(f"Saving model at epoch {epoch}")
                checkpoint_name = self.save_dir+self.save_name+f"_{epoch}"
                self.model.save_model(checkpoint_name)
                if save_plots:
                    y_pred = jax.vmap(self.model, in_axes=(None, 0, 0))(ts, yi[:,0,:], ui)
                    self.plot_training(ts, yi, y_pred, ui, epoch, train_loss_epoch)

        if print_status: print("Training complete.")
        checkpoint_name = self.save_dir+self.save_name+"_final"
        if print_status: print(f"Saving model at {checkpoint_name}")
        self.model.save_model(checkpoint_name)
        if save_plots:
            self.plot_training(ts, yi, y_pred, ui, "final", train_loss_epoch)

        return self.model, train_losses, valid_losses
    
    def plot_training(self, ts, yi, y_pred, ui, epoch, train_loss_epoch):
        fig_states, ax_states = plt.subplots(figsize=(10, 5))

        for i in range(yi.shape[-1]):
            line_true, = ax_states.plot(ts, yi[0, :, i],           label=f"True {i}")
            ax_states.plot(ts, y_pred[0, :, i], "--", color=line_true.get_color(), label=f"Predicted {i}")

        ax_states.set_xlabel("Time")
        ax_states.set_ylabel("State")
        ax_states.grid(which="both")
        #ax_states.legend(loc="best")
        fig_states.suptitle(f"Epoch {epoch} – Training Loss: {train_loss_epoch:.4e}")
        fig_states.tight_layout()

        fname_states = f"{self.save_dir}png/epoch_{epoch}_states_loss_{train_loss_epoch:.4e}.png"
        fig_states.savefig(fname_states, dpi=150)
        plt.close(fig_states)         # no display

        fig_field, ax_field = plt.subplots(figsize=(10, 5))

        ax_field.plot(ts, ui[0, :, ])

        ax_field.set_xlabel("Time")
        ax_field.set_ylabel("E-field")
        ax_field.grid(which="both")
        fig_states.suptitle(f"Epoch {epoch} – Training Loss: {train_loss_epoch:.4e}")
        fig_states.tight_layout()

        fname_field = f"{self.save_dir}png/epoch_{epoch}_field_loss_{train_loss_epoch:.4e}.png"
        fig_field.savefig(fname_field, dpi=150)
        plt.close(fig_field)         # no display