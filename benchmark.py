import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from timeit import timeit
from tqdm import tqdm

from online_softmax import (
    online_softmax,
    online_softmax_dot,
    naive_online_softmax_dot,
    naive_softmax_dot,
)

bench1 = """
key = jax.random.PRNGKey({N})
key, subkey = jax.random.split(key)
x = jax.random.normal(key, ({M}, {N}))
#x = jnp.asarray(np.random.randn({M}, {N}))

def bench():
    return online_softmax(x, 1).block_until_ready()
"""

bench2 = """
key = jax.random.PRNGKey({N})
key, subkey = jax.random.split(key)
x = jax.random.normal(key, ({M}, {N}))
#x = jnp.asarray(np.random.randn({M}, {N}))

def bench():
    return jax_softmax(x, 1).block_until_ready()
"""

jax_softmax = jax.jit(jax.nn.softmax, static_argnames=("axis",))

def benchmark_online_softmax(batch_size=16):
    runs1, runs2 = [], []
    n = 10000

    _globals = {'jax': jax, 'jnp': jnp, 'np': np}
    x = np.concatenate([np.array([4]), 2**jnp.arange(2, 15)])
    for N in tqdm(x):
        t1 = timeit("bench()",
            setup=bench1.format(M=batch_size, N=N),
            globals={**_globals, 'online_softmax': online_softmax}, number=n)
        runs1.append(t1 / n)

    for N in tqdm(x):
        t2 = timeit("bench()",
            setup=bench2.format(M=batch_size, N=N),
            globals={**_globals, 'jax_softmax': jax_softmax}, number=n)
        runs2.append(t2 / n)
    return {
        'M': batch_size, 'N': x[1:],
        "online_softmax": np.asarray(runs1)[1:],    # first iteration is skipped for jit tracing time
        "jax_softmax": np.asarray(runs2)[1:]}

def plot_online_softmax(device_name, batch_size=32):
    result = benchmark_online_softmax(batch_size)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(result['N'], result['jax_softmax']*1000, label='jax softmax', marker='*')
    ax.plot(result['N'], result['online_softmax']*1000, label='online softmax', marker='*')
    ax.legend()
    ax.set_title(f"batch size={batch_size}, device={device_name}")
    ax.set_xlabel('vector size')
    ax.set_ylabel('runtime (ms)')
    ax.set_xscale('log')
    return fig


bench3 = """
x, y = jnp.asarray(np.random.randn(2, {M}, {N}))

def bench():
    online_softmax_dot(x, y, 1).block_until_ready()
"""

bench4 = """
x, y = jnp.asarray(np.random.randn(2, {M}, {N}))

def bench():
    naive_online_softmax_dot(x, y, 1).block_until_ready()
"""

bench5 = """
x, y = jnp.asarray(np.random.randn(2, {M}, {N}))

def bench():
    naive_softmax_dot(x, y, 1).block_until_ready()
"""

def benchmark_online_softmax_dot(batch_size=32):
    runs1, runs2, runs3 = [], [], []
    n = 1000*10

    key = jax.random.PRNGKey(42)
    _globals = {'key': key, 'jax': jax, 'jnp': jnp, 'np': np}
    x = np.concatenate([np.array([4]), 2**jnp.arange(2, 15)])
    np.random.seed(42)
    for N in tqdm(x):
        t1 = timeit(
            "bench()",
            setup=bench3.format(M=batch_size, N=N),
            globals={**_globals, 'online_softmax_dot': online_softmax_dot},
            number=n)
        runs1.append(t1 / n)

    np.random.seed(42)
    for N in tqdm(x):
        t2 = timeit(
            "bench()",
            setup=bench4.format(M=batch_size, N=N),
            globals={**_globals, 'naive_online_softmax_dot': naive_online_softmax_dot},
            number=n)
        runs2.append(t2 / n)

    np.random.seed(42)
    for N in tqdm(x):
        t3 = timeit(
            "bench()",
            setup=bench5.format(M=batch_size, N=N),
            globals={**_globals, 'naive_softmax_dot': naive_softmax_dot},
            number=n)
        runs3.append(t3 / n)
    return {
        'M': batch_size, 'N': x[1:],
        "online_softmax_dot": np.asarray(runs1)[1:],    # first iteration is skipped for jit tracing time
        "naive_online_softmax_dot": np.asarray(runs2)[1:],
        "naive_softmax_dot": np.asarray(runs3)[1:]}


def plot_online_softmax_dot(device_name, batch_size=16):
    result = benchmark_online_softmax_dot(batch_size)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(result['N'], result['naive_softmax_dot']*1000, label='naive_softmax_dot', marker='*')
    ax.plot(result['N'], result['online_softmax_dot']*1000, label='online_softmax_dot', marker='*')
    ax.plot(result['N'], result['naive_online_softmax_dot']*1000, label='naive_online_softmax_dot', marker='*')
    ax.legend()
    ax.set_title(f"Batch size={batch_size}, device={device_name}")
    ax.set_xlabel('vector size')
    ax.set_ylabel('runtime (ms)')
    ax.set_xscale('log')
    return fig


if __name__ == "__main__":
    plot_online_softmax("GTX 1080 Ti", 16)
    plt.savefig('assets/online_softmax.png')
    plot_online_softmax_dot("GTX 1080 Ti", 16)
    plt.savefig('assets/online_softmax_dot.png')