import jax
import jax.numpy as jnp
from rom import LDSModel
from train import Trainer
import pickle

with open("data/dataset.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/modes.pkl", "rb") as f:
    modes = pickle.load(f)

with open("data/params.pkl", "rb") as f:
    params = pickle.load(f)

data = jnp.array(data)
modes = jnp.array(modes)
ts = jnp.array(params.ts)

K = data.shape[0]
assert K == modes.shape[0]

print(data.shape)
print(modes.shape)

key = jax.random.key(42)
rom = LDSModel(modes.shape[-1], data.shape[-1], 1e-3, key=key)

print(rom(ts[:100],data[0,0,:],modes[0,:100,:]).shape)

trainer = Trainer(model=rom)

rom, train_losses, _ = trainer.train(
    ts=ts, 
    data_in=modes,
    data_out=data, 
    n_epochs=20000, 
    bs=32, 
    time_windows=[0.1,5,5], 
    n0=0, 
    nrand=None, 
    save_every=100, 
    seed=0, 
    print_status=True, 
    save_plots=True, 
    permute=True)