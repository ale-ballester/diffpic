import jax
import jax.numpy as jnp

class DataLoader: 
    def __init__(self, ts, data_in, data_out, permute=True): 
        self.ts = ts 
        self.data_in = data_in
        self.data_out = data_out
        self.n_times = len(ts) 
        self.permute = permute
        
    def __call__(self, batch_size, key, ind_times=None, n0=None, n1=None, nskip=None, nobs=None, nrand=None): 
        key_perm, key_times = jax.random.split(key, 2) 
        dataset_size = self.data_in.shape[0] 
        indices = jnp.arange(dataset_size) 
        #if n1 is None: n1 = 
        if nrand is None and n0 is not None and n1 is not None and nskip is not None: 
            if ind_times is None: ind_times = slice(n0, n1, nskip) 
        while True: 
            if self.permute:
                perm = jax.random.permutation(key_perm, indices)
                #print("Permuted indices: ", perm)
            (key_perm,) = jax.random.split(key_perm, 1) 
            start = 0 
            end = batch_size 
            while start <= dataset_size - batch_size:
                end = int(min(start + batch_size, dataset_size))
                if n0 is None: 
                    key_times,key_n0 = jax.random.split(key_times, 2) 
                    n0 = jax.random.choice(key_n0, jnp.arange(self.n_times-n1)) if n1 < self.n_times else 0
                    #print(f"Random start time index: {n0}")
                else: 
                    (key_times,) = jax.random.split(key_times, 1)
                    n0 = n0
                if nrand is not None: 
                    win_len = min(n1, self.n_times-n0)
                    nrand_local = min(nrand, win_len)
                    #print(f"Random time window: [{n0}, {n0+win_len}]")
                    #print(f"Selecting {nrand_local} random times from {len(jnp.arange(n0,n0+win_len))} available.")
                    ind_times = jnp.sort(jax.random.choice(key_times, jnp.arange(n0,n0+win_len), shape=(nrand_local,), replace=False))
                    #print("Selected time indices (length): ", len(ind_times))
                    #print("Selected time indices: ", ind_times)
                else:
                    win_len = min(n1, self.n_times-n0)
                    ind_times = jnp.arange(n0,n0+win_len)
                if self.permute:
                    #print(f"Using permutation for batching from {start} to {end}.")
                    batch_perm = perm[start:end]
                else:
                    #print("Using sampling with replacement for batching.")
                    batch_perm = jax.random.choice(key_perm, indices, shape=(batch_size,), replace=True)
                    (key_perm,) = jax.random.split(key_perm, 1)
                #print("Batch indices: ", batch_perm)
                #print("Time indices: ", ind_times)
                subset_in = self.data_in[batch_perm,:,:] 
                subset_in = subset_in[:,ind_times,:]
                subset_out = self.data_out[batch_perm,:,:] 
                subset_out = subset_out[:,ind_times,:]
                #subset = subset[:,:,nobs] 
                yield self.ts[ind_times], subset_in, subset_out
                start = end