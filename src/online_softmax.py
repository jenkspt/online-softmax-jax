from typing import NamedTuple
from functools import partial
import jax
import jax.numpy as jnp


class OnlineSoftmaxState(NamedTuple):
    max: jnp.ndarray
    scale: jnp.ndarray


@partial(jax.jit, static_argnames=("axis",))
def online_softmax(x, axis):
    def op(a: OnlineSoftmaxState, b: OnlineSoftmaxState) -> OnlineSoftmaxState:
        _max = jnp.maximum(a.max, b.max)
        return OnlineSoftmaxState(_max,
            a.scale * jnp.exp(a.max - _max) + b.scale * jnp.exp(b.max - _max))

    state = jax.lax.reduce(
        OnlineSoftmaxState(x, jnp.ones_like(x)),
        OnlineSoftmaxState(jnp.full_like(x, -jnp.inf, shape=()), jnp.zeros_like(x, shape=())),
        op, (axis,)
    )
    x_max = jax.lax.stop_gradient(jnp.expand_dims(state.max, axis))
    scale = jnp.expand_dims(state.scale, axis)
    return jnp.exp(x - x_max) / scale


class SoftmaxDotState(NamedTuple):
    max: jnp.ndarray
    scale: jnp.ndarray
    total: jnp.ndarray


@partial(jax.jit, static_argnames=("axis",))
def online_softmax_dot(x, y, axis):

    def op(a: SoftmaxDotState, b: SoftmaxDotState) -> SoftmaxDotState:
        _max = jnp.maximum(a.max, b.max)
        beta1 = jnp.exp(a.max - _max)
        beta2 = jnp.exp(b.max - _max)
        return SoftmaxDotState(
            _max,
            a.scale * beta1 + b.scale * beta2,
            a.total * beta1 + b.total * beta2)

    state = jax.lax.reduce(
        SoftmaxDotState(x, jnp.ones_like(x), y),
        SoftmaxDotState(
            jnp.full_like(x, -jnp.inf, shape=()),
            jnp.zeros_like(x, shape=()),
            jnp.zeros_like(y, shape=())),
        op, (axis,)
    )

    return state.total / state.scale


@partial(jax.jit, static_argnames=("axis",))
def naive_softmax_dot(x, y, axis):
    return (jax.nn.softmax(x, axis) * y).sum(axis)


@partial(jax.jit, static_argnames=("axis",))
def naive_online_softmax_dot(x, y, axis):
    return (online_softmax(x, axis) * y).sum(axis)
