import jax
import jraph
import jax.numpy as jnp
import haiku as hk
from typing import Tuple
from models.utils import GraphNetwork, replace_node_features

class GraphNVPLayer(hk.Module):
    def __init__(self, dim, mask_dim, hidden_dim = 16, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        edge_output_sizes = [hidden_dim, hidden_dim]
        node_output_sizes_trans = [hidden_dim, hidden_dim, 1]
        node_output_sizes_scale = [hidden_dim, hidden_dim, dim]
        def make_mlp_edge_update(activation):
            @jraph.concatenated_args
            def f(feats):
                return hk.nets.MLP(edge_output_sizes, activation=activation)(feats)
            return f
        def make_mlp_node_update(activation, node_output_sizes):
            def f(node_feats, sender_feats, receiver_feats, global_feats):
                return hk.nets.MLP(node_output_sizes, activation=activation)(
                    jnp.concatenate([node_feats, receiver_feats], axis=1) # only aggr over msgs from incoming edges
                )
            return f
        self.mp_trans = GraphNetwork(update_edge_fn=make_mlp_edge_update(jax.nn.relu), 
                                     update_node_fn=make_mlp_node_update(jax.nn.relu, 
                                                                         node_output_sizes_trans), 
                                     update_global_fn=None)
        self.mp_scale = GraphNetwork(update_edge_fn=make_mlp_edge_update(jax.nn.relu), 
                                     update_node_fn=make_mlp_node_update(jax.nn.tanh,
                                                                         node_output_sizes_scale), 
                                     update_global_fn=None)
    def mask_graph(self, g):
        nodes = g.nodes
        mask = jnp.ones_like(nodes).at[:, self.mask_dim].set(0)
        g_masked = replace_node_features(g, nodes * mask)
        mask_comp = 1-mask
        return g_masked, mask_comp
    
    def forward(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        g_masked, mask_comp = self.mask_graph(g)
        scale, trans = self.mp_scale(g_masked).nodes, self.mp_trans(g_masked).nodes
        new_nodes = g.nodes * jnp.exp(scale * mask_comp) + (trans * mask_comp)
        logdetJ = jnp.sum(scale * mask_comp)
        g_new = replace_node_features(g, new_nodes)
        return g_new, logdetJ
        
    def reverse(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        g_masked, mask_comp = self.mask_graph(g)
        scale, trans = self.mp_scale(g_masked).nodes, self.mp_trans(g_masked).nodes
        new_nodes = (g.nodes - (trans * mask_comp)) / jnp.exp(scale * mask_comp)
        logdetJ = -jnp.sum(scale * mask_comp)
        g_new = replace_node_features(g, new_nodes)
        return g_new, logdetJ
    
class GraphNVPBlock(hk.Module):
    def __init__(self, dim, hidden_dim = 16, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.layers = [GraphNVPLayer(dim, mask_dim, hidden_dim) for mask_dim in range(dim)]
        
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
    
class GraphNVP(hk.Module):
    # The potential advantage of GraphNVP over GRevNet is that
    # 1. This handles edge feature updates but the latter couldn't
    # 2. The latter requires breaking node features into two halves which makes
    # application to 3D coordinate as node features difficult. GraphNVP
    # can iterate and update over each dimension using the rest features 
    # (see paper for more details) 
    def __init__(self, n_layers, dim, hidden_dim=16, name=None):
        super().__init__(name=name)
        self.blocks = [GraphNVPBlock(dim, hidden_dim) for _ in range(n_layers)]
    
    # these should be same as before, just copied
    def forward(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        ldj_sum = 0
        for block in self.blocks:
            g, ldj = block.forward(g)
            ldj_sum += ldj
        return g, ldj_sum
        
    def reverse(self, g: jraph.GraphsTuple) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        ldj_sum = 0
        for block in self.blocks[::-1]:
            g, ldj = block.reverse(g)
            ldj_sum += ldj
        return g, ldj_sum