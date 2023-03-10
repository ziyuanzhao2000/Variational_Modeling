import jax.numpy as jnp
import optax

def gaussian_kernel(x, y, s=1., axis=-1):
    norm_sq = jnp.sum((x-y)*(x-y), axis=axis)
    return jnp.exp(-norm_sq/(2*s**2))

def gaussian_kernel_mean(x, y, s=1., axis=-1):
    norm_sq = jnp.mean((x-y)*(x-y), axis=axis)
    return jnp.exp(-norm_sq/(2*s**2))
    
def compute_kernel(x, y, kernel=gaussian_kernel_mean):
    x_size = x.shape[0]
    y_size = y.shape[0]
    x = jnp.expand_dims(x, 1) # (x_size, 1, dim)
    y = jnp.expand_dims(y, 0) # (1, y_size, dim)
    tiled_x = jnp.tile(x, (1, y_size, 1))
    tiled_y = jnp.tile(y, (x_size, 1, 1))
    return kernel(tiled_x, tiled_y, axis=2)

def mmd(x, y, k=gaussian_kernel_mean):
    """
    x: N * d array where each row is a sample from pdf p(.)
    y: N * d array where each row is a sample from pdf q(.)
    k: kernel function that measures similarity of u / v
    """
    N = x.shape[0]
    
    kxixj = compute_kernel(x, x, k)
    kyiyj = compute_kernel(y, y, k)
    kxiyj = compute_kernel(x, y, k)
    kyixj = compute_kernel(y, x, k)

    return jnp.sum(jnp.triu(kxixj + kyiyj - kxiyj - kyixj, 1)) / (N*(N-1)/2)

def forward_KL(net_rev, params, x_batch): # x.shape = n_batch, 2 * n_channels
    loss = 0
    batch_size = x_batch.shape[0]
    batch_out, ldj = net_rev.apply(params, x_batch)
    loss += -jnp.sum(optax.l2_loss(batch_out.nodes)) # assume iid standard gaussian dist on node features
    loss += ldj
    #jax.debug.print(f"LDJ: {ldj}")
    return -loss / batch_size

def reverse_KL(net_fwd, en_template, params, z_batch, T=1):
    """
    Currently we do not support dynamically shaped graphs
    """
    loss = 0
    batch_size = z_batch.shape[0]
    batch_out, ldj = net_fwd.apply(params, z_batch)
    batch_nodes = batch_out.nodes
    total_energy = 0
    for nodes in jnp.vsplit(batch_nodes, batch_size):
        nodes = training_PCA_inverse_transform(nodes.reshape(1, -1)).reshape(-1, 2)
        total_energy += en_template.energy(params=nodes)

    loss -= 1/T * total_energy 
    loss += ldj # we ignore log p_u term because it's not affected by model param
    return -loss / batch_size