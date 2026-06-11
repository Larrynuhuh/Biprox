import jax
import jax.numpy as jnp
import time
from main import solve

parallel_solve = jax.jit(jax.vmap(solve, in_axes=(0, None, None)), static_argnames='terms')

# ==========================================
# TEST 1: The Weird Negative Exponent Test
# ==========================================
print("--- TEST 1: Negative Exponent ---")
weird_ans = solve(x=0.002, n=-0.5, terms=50)
print(f"JAX Result for (1 + 0.002)^-0.5: {weird_ans}")
print(f"True Math Calculator Result:    {1 / (1.002**0.5)}\n")


# ==========================================
# TEST 2 & BENCHMARK: Massive Workload
# ==========================================
print("--- TEST 2: Benchmarking Massive Parallel Workload ---")

# Generate 10,000 random x values between 0.001 and 0.5 (safely less than 1)
key = jax.random.PRNGKey(42)
massive_x = jax.random.uniform(key, shape=(10000,), minval=0.001, maxval=0.5)

# WARMUP RUN (Important for JAX! This compiles the code so compilation time doesn't skew our benchmark)
_ = parallel_solve(massive_x, 0.5, 100).block_until_ready()

# THE REAL BENCHMARK RUN
# We are calculating 100 terms of the theorem for 10,000 different numbers simultaneously!
start_time = time.perf_counter()

results = parallel_solve(massive_x, 0.5, 100).block_until_ready()

end_time = time.perf_counter()

execution_time = end_time - start_time
print(f"Successfully processed {len(results)} distinct math problems.")
print(f"Total Terms calculated: {len(results) * 100:,} steps.")
print(f"⏱️ Execution Time: {execution_time * 1000:.4f} milliseconds")