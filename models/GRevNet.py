import jax
import jraph
import jax.numpy as jnp
import haiku as hk
from typing import Tuple
from models.utils import GraphNetwork, replace_node_features

class GRevHalfLayer(hk.Module):
    def __init__(self, dim, hidden_dim = 16, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.half_dim = dim // 2
        assert dim % 2 == 0
        edge_output_sizes = [hidden_dim, hidden_dim, dim]
        node_output_sizes = [hidden_dim, hidden_dim, self.half_dim]
        # The MLP setup is for initial testing, and it works, 
        # but we will use MPNN since we work with graphs
        #self.mp_trans = hk.nets.MLP(output_sizes)  
        #self.mp_scale = hk.nets.MLP(output_sizes)
        def make_mlp_edge_update(activation):
            @jraph.concatenated_args
            def f(feats):
                return hk.nets.MLP(edge_output_sizes, activation=activation)(feats)
            return f
        # could have written lambda but I don't want to be that terse
        def make_mlp_node_update(activation):
            def f(node_feats, sender_feats, receiver_feats, global_feats):
                return hk.nets.MLP(node_output_sizes, activation=activation)(
                    jnp.concatenate([node_feats, receiver_feats], axis=1) # only aggr over msgs from incoming edges
                )
            return f
        # note, floating point error can accum at current precision
        self.mp_trans = GraphNetwork(update_edge_fn=make_mlp_edge_update(jax.nn.relu), 
                                     update_node_fn=make_mlp_node_update(jax.nn.relu), 
                                     update_global_fn=None)
        self.mp_scale = GraphNetwork(update_edge_fn=make_mlp_edge_update(jax.nn.relu), 
                                     update_node_fn=make_mlp_node_update(jax.nn.tanh), 
                                     update_global_fn=None)
        
    def forward(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        half_feats_0 = g.nodes[:, :self.half_dim]
        half_feats_1 = g.nodes[:, self.half_dim:]
        half_g_1 = replace_node_features(g, half_feats_1)
        scale, trans = self.mp_scale(half_g_1).nodes, self.mp_trans(half_g_1).nodes
        new_half_feats_0 = half_feats_0 * jnp.exp(scale) + trans
        new_nodes = g.nodes.at[:, :self.half_dim].set(new_half_feats_0)
        logdetJ = jnp.sum(scale)
        g_new = replace_node_features(g, new_nodes)
        return g_new, logdetJ
    
    def reverse(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        half_feats_0 = g.nodes[:, :self.half_dim]
        half_feats_1 = g.nodes[:, self.half_dim:]
        half_g_1 = replace_node_features(g, half_feats_1)
        scale, trans = self.mp_scale(half_g_1).nodes, self.mp_trans(half_g_1).nodes
        new_half_feats_0 = (half_feats_0 - trans) / jnp.exp(scale)
        new_nodes = g.nodes.at[:, :self.half_dim].set(new_half_feats_0)
        logdetJ = -jnp.sum(scale)
        g_new = replace_node_features(g, new_nodes)
        return g_new, logdetJ


# we put two half-layers together serially, but switching first and second half of 
# edge features in the middle, see the figure below this code block for viz.
class GRevLayer(hk.Module):
    def __init__(self, dim, hidden_dim = 16, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.half_dim = dim // 2
        self.half_layer1 = GRevHalfLayer(dim, hidden_dim)
        self.half_layer2 = GRevHalfLayer(dim, hidden_dim)
    
    def swap(self, g: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return replace_node_features(g, jnp.concatenate([g.nodes[:, self.half_dim:], 
                                                         g.nodes[:, :self.half_dim]], axis=1))
        
    def forward(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        g, ldj1 = self.half_layer1.forward(g) 
        g = self.swap(g)
        g, ldj2 = self.half_layer2.forward(g)
        return g, ldj1+ldj2
        
    def reverse(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        g, ldj1 = self.half_layer2.reverse(g)
        g = self.swap(g)
        g, ldj2 = self.half_layer1.reverse(g)
        return g, ldj1+ldj2
    

class GRevNet(hk.Module):
    def __init__(self, n_layers, dim, hidden_dim=16, name=None):
        super().__init__(name=name)
        self.layers = [GRevLayer(dim, hidden_dim) for _ in range(n_layers)]
        
    def forward(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        ldj_sum = 0
        for layer in self.layers:
            g, ldj = layer.forward(g)
            ldj_sum += ldj
        return g, ldj_sum
        
    def reverse(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        ldj_sum = 0
        for layer in self.layers[::-1]:
            g, ldj = layer.reverse(g)
            ldj_sum += ldj
        return g, ldj_sum