import haiku as hk
import jraph
import jax.numpy as jnp

def replace_node_features(g: jraph.GraphsTuple, new_nodes) -> jraph.GraphsTuple:
    nodes, edges, receivers, senders, globals_, n_node, n_edge = g
    n_node = jnp.array([len(nodes)])
    return jraph.GraphsTuple(new_nodes, edges, receivers, senders, globals_, n_node, n_edge)


class GraphNetwork(hk.Module):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.model = jraph.GraphNetwork(*args, **kwargs)
    
    def __call__(self, g: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self.model(g)