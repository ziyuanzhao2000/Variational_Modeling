# 3/7/22 test a distrax based RealNVP model
from typing import Sequence, Tuple, Callable
import jax
import jax.numpy as jnp
Array = jnp.array
import haiku as hk
import numpy as np
import distrax
    
def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int]) -> hk.Sequential:
    log_scaler = hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)), # so flatten all event dimensions
        hk.nets.MLP(hidden_sizes, activation=jax.nn.tanh), # core MLP
        hk.Linear(
          np.prod(event_shape),
          w_init=jnp.zeros,
          b_init=jnp.zeros),
        hk.Reshape(event_shape, preserve_dims=-1), # unflatten the last dim to event shape
    ])
    
    shifter = hk.Sequential([
        hk.Flatten(preserve_dims=-len(event_shape)),
        hk.nets.MLP(hidden_sizes, activation=jax.nn.relu), # use relu
        hk.Linear(
          np.prod(event_shape),
          w_init=jnp.zeros,
          b_init=jnp.zeros),
        hk.Reshape(event_shape, preserve_dims=-1), 
    ])
    
    def conditioner(batch):
        return (shifter(batch), log_scaler(batch))
    
    return conditioner
    
    
# affine bijector as used in RealNVP
def bijector_fn(params: Tuple):
    shift, log_scale = params
    return distrax.ScalarAffine(shift=shift,
                               log_scale=log_scale)


def make_RealNVP_flow(num_layers = 8,
                   event_shape = [2],
                   hidden_sizes = [4, 4, 2]) -> Tuple[distrax.Transformed, distrax.Bijector]:

    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes)
        )
        layers.append(layer)
        mask = jnp.logical_not(mask) 
    
    flow = distrax.Chain(layers)
    return flow

def make_nway_affine_coupling_flow(num_layers = 8,
                                  event_shape = [3], # can be even/odd!
                                  coupling_dim = 0, 
                                  hidden_sizes = [4, 4, 2]):
    # each submask responsible for masking a slice along the coupling dim
    n_mask = event_shape[coupling_dim]
    assert num_layers % n_mask == 0
    masks = jnp.zeros((n_mask,) + tuple(event_shape))
    for i in range(n_mask):
        masks = masks.at[(i,) + (slice(None),) * coupling_dim + (i,) + \
                         (slice(None),) * (len(event_shape) - 1)].set(1)
    masks = masks.astype(bool)


    layers = []
    for i in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=masks[i % n_mask],
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes)
        )
        layers.append(layer)
    
    flow = distrax.Chain(layers)
    return flow

def make_grouped_affine_coupling_flow(num_layers = 8,
                                  event_shape = [3], # can be even/odd!
                                  coupling_dim = 0, 
                                  coupling_group_size = 1,
                                  hidden_sizes = [4, 4, 2]):
    # each submask responsible for masking a slice along the coupling dim
    n_mask = event_shape[coupling_dim]
    assert n_mask % coupling_group_size == 0
    n_mask = n_mask // coupling_group_size
    assert num_layers % n_mask == 0

    masks = jnp.zeros((n_mask,) + tuple(event_shape))
    for i in range(n_mask):
        for offset in range(coupling_group_size):
            masks = masks.at[(i,) + (slice(None),) * coupling_dim + (i+offset,) + \
                            (slice(None),) * (len(event_shape) - 1)].set(1)
    masks = masks.astype(bool)

    layers = []
    for i in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=masks[i % n_mask],
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes)
        )
        layers.append(layer)
    
    flow = distrax.Chain(layers)
    return flow

def make_forward_reverse_flow_models(make_flow_model: Callable,
                                     *args, **kwargs):

    @hk.without_apply_rng
    @hk.transform
    def reverse_model(batch): # x -> z
        flow = make_flow_model(*args, **kwargs)
        z, ld = flow.inverse_and_log_det(batch)
        return z, ld

    @hk.without_apply_rng
    @hk.transform
    def forward_model(batch):
        flow = make_flow_model(*args, **kwargs)
        x, ld = flow.forward_and_log_det(batch)
        return x, ld
    
    return forward_model, reverse_model

def make_base_dist(event_shape):
    base_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros(shape=event_shape),
                                                   scale_diag=jnp.ones(shape=event_shape))
    return base_dist

def make_KL_loss_funs(forward_model, reverse_model, energy_fun, base_dist):
    def forward_KL(params, batch):
        z, ld = reverse_model.apply(params, batch)
        loss = -jnp.mean(ld)
        loss -= jnp.mean(base_dist.log_prob(z))
        return loss

    def reverse_KL(params, batch, beta=1):
        loss = 0
        x, ld = forward_model.apply(params, batch)
        loss -= jnp.mean(ld)
        loss += jnp.mean(beta*energy_fun(x))
        return loss
        
    def combined_KL(params, x_batch, z_batch, beta=1):
        loss = 0.2*forward_KL(params, x_batch) + 0.8*reverse_KL(params, z_batch, beta)
        return loss
    
    return forward_KL, reverse_KL, combined_KL