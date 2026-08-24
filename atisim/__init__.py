"""JAX 6-DOF fixed-wing flight dynamics.

float64 is enabled here, before any JAX array is created, because it must be set
before the first array allocation to take effect. float32 is marginal for the
Newton trim solve and for quaternion norm stability over long rollouts.
"""

import jax

jax.config.update("jax_enable_x64", True)
