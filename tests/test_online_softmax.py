import jax
import jax.numpy as jnp

from online_softmax import (
    online_softmax,
    online_softmax_dot,
    naive_softmax_dot,
)


def test_online_softmax():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (4, 8))
    a, b = online_softmax(x, 0), jax.nn.softmax(x, 0)
    assert a.shape == b.shape
    assert jnp.allclose(a, b)

    a, b = online_softmax(x, 1), jax.nn.softmax(x, 1)
    assert a.shape == b.shape
    assert jnp.allclose(a, b)


def test_softmax_dot():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x, y = jax.random.normal(subkey, (2, 4, 8))
    a, b = online_softmax_dot(x, y, 0), naive_softmax_dot(x, y, 0)
    assert a.shape == b.shape
    assert jnp.allclose(a, b)

    a, b = online_softmax_dot(x, y, 1), naive_softmax_dot(x, y, 1)
    assert a.shape == b.shape
    assert jnp.allclose(a, b)