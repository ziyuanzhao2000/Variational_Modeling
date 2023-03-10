import jax
import jax.numpy as jnp
import haiku as hk

class RealNVPLayer(hk.Module):
    def __init__(self, n_channels, n_hidden, name=None):
        super().__init__(name)
        self.S = hk.nets.MLP(output_sizes=[n_hidden, n_hidden, n_hidden, n_channels], activation=jax.nn.tanh)
        self.T = hk.nets.MLP(output_sizes=[n_hidden, n_hidden, n_hidden, n_channels], activation=jax.nn.relu)
    
    def forward(self, z1, z2):
        x1 = z1
        sz1 = self.S(z1)
        x2 = (z2 - self.T(x1)) * jnp.exp(-sz1)
        logdet = -jnp.sum(sz1)
        return x1, x2, logdet 
    
    def reverse(self, x1, x2):
        z1 = x1
        sx1 = self.S(x1)
        z2 = x2 * jnp.exp(sx1) + self.T(x1)
        logdet = jnp.sum(sx1)
        return z1, z2, logdet

class RealNVPBlock(hk.Module):
    def __init__(self, n_channels, n_hidden, name=None):
        super().__init__(name)
        self.layer1 = RealNVPLayer(n_channels, n_hidden)
        self.layer2 = RealNVPLayer(n_channels, n_hidden)
    
    def forward(self, z1, z2):
        y1, y2, ld1 = self.layer1.forward(z1, z2)
        x1, x2, ld2 = self.layer2.forward(y2, y1) # swap channels
        logdet = ld1 + ld2
        return x1, x2, logdet
    
    def reverse(self, x1, x2):
        y1, y2, ld1 = self.layer2.reverse(x1, x2)
        z1, z2, ld2 = self.layer1.reverse(y2, y1) # swap channels
        logdet = ld1 + ld2
        return z1, z2, logdet

class RealNVPStack(hk.Module):
    def __init__(self, n_channels, n_stacks, n_hidden, name=None):
        super().__init__(name)
        self.layers = [RealNVPBlock(n_channels, n_hidden) for _ in range(n_stacks)]
    
    def forward(self, z1, z2):
        logdet = 0
        for layer in self.layers:
            z1, z2, ld = layer.forward(z1, z2)
            logdet += ld
        return z1, z2, logdet
    
    def reverse(self, x1, x2):
        logdet = 0
        for layer in self.layers[::-1]:
            x1, x2, ld = layer.reverse(x1, x2)
            logdet += ld
        return x1, x2, logdet
    