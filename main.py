import jax
import jax.numpy as jnp

@jax.jit(static_argnames='terms')
def solve(x, n, terms):
    r = jnp.arange(1, terms+1)
    coeff = (n - r + 1)/r
    term = coeff * x

    prod_term = jnp.concatenate([jnp.array([1]), jnp.cumprod(term)])
    summed_term = jnp.sum(prod_term)

    return summed_term

