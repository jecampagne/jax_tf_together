# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: jaxccl
#     language: python
#     name: jaxccl
# ---

# +
from typing import Callable
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.experimental import host_callback as hcb
import numpy as np

import tensorflow as tf
# -

jax.__version__

tf.__version__


# +
def call_tf_no_ad(tf_fun: Callable, arg, *, result_shape):
  """The simplest implementation of calling to TF, without AD support.
  We must use hcb.call because the TF invocation must happen outside the
  JAX staged computation."""

  def tf_to_numpy(t):
    # Turn the Tensor to NumPy array without copying.
    return np.asarray(memoryview(t)) if isinstance(t, tf.Tensor) else t

  return hcb.call(lambda arg: tf.nest.map_structure(tf_to_numpy,
                                                    tf_fun(arg)),
                  arg, result_shape=result_shape)


def call_tf_full_ad(tf_fun: Callable, arg, *, result_shape):
  """Calls a TensorFlow function with support for reverse AD.
  Supports higher-order AD and pytree arguments.
  """

  @jax.custom_vjp
  def make_call(arg):
    """We wrap it all in `make_call` so that we can attach custom VJP."""
    return call_tf_no_ad(tf_fun, arg, result_shape=result_shape)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    return make_call(arg), arg  # Return the primal argument as the residual

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def tf_vjp_fun(arg_and_ct_res):
      """Invoke TF gradient; used with hcb.call."""
      arg, ct_res = arg_and_ct_res

      def make_var(a):
        return a if isinstance(a, tf.Variable) else tf.Variable(a)

      arg_var = tf.nest.map_structure(make_var, arg)

      with tf.GradientTape(persistent=True) as tape:
        res = tf_fun(arg_var)

      tf.nest.assert_same_structure(res, ct_res)
      accumulator = None  # Accumulate argument cotangent. Same structure as "arg"

      def acc_ct(res_, ct_res_):
        dres_darg = tape.gradient(res_, sources=arg_var,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        tf.nest.assert_same_structure(dres_darg, arg)
        scaled_dres_darg = tf.nest.map_structure(lambda d: d * ct_res_, dres_darg)
        nonlocal accumulator
        accumulator = (scaled_dres_darg if accumulator is None
                       else tf.nest.map_structure(lambda x, y: x + y,
                                                  accumulator, scaled_dres_darg))

      tf.nest.map_structure(acc_ct, res, ct_res)
      return accumulator

    return (call_tf_full_ad(tf_vjp_fun, (arg, ct_res),
                            result_shape=arg),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)


# -

def f(x):
  return call_tf_full_ad(tf.math.sin, 2. * x, result_shape=x)


x_test=np.pi/5.

f(x_test), np.sin(2*x_test)

jax.grad(f)(x_test), 2*np.cos(2*x_test)

import cosmopower as cp # Cosmo Power (from Alessio Spurio Mancini since 10thJuly22)

# +
cp_dir = "./cosmo_power_trained/"
# instantiate your CP linear power emulator

pklin_cp = cp.cosmopower_NN(restore=True,
                            restore_filename=cp_dir+'/PKLIN_NN') # change with path to your linear power emulator .pkl file, without .pkl suffix


# instantiate your CP nonlinear correction (halofit) emulator 
pknlratio_cp = cp.cosmopower_NN(restore=True,
                                restore_filename=cp_dir+'/PKBOOST_NN') # change with path to your nonlinear correction emulator .pkl file, without .pkl suffix
# -

h_emu = 0.6774 
Omega_c_emu = 0.2589
Omega_b_emu = 0.0486
sigma8_emu = 0.8159
n_s_emu = 0.9667
z_test = 2.0

params_cosmo_power = {'Omega_cdm': [Omega_c_emu],
              'Omega_b':   [Omega_b_emu],
              'h':         [h_emu],
              'n_s':       [n_s_emu],
              'sigma8':    [sigma8_emu],
             }
params_cosmo_power['z'] = [z_test]


def tf_test_cp(params):
    parameters_arr = pklin_cp.dict_to_ordered_arr_np(params)
    x_tensor = tf.convert_to_tensor(parameters_arr, dtype=tf.float32)
    return 10.**(pklin_cp.predictions_tf(x_tensor)+pknlratio_cp.predictions_tf(x_tensor))


tf_pk = tf_test_cp(params_cosmo_power)

tf_pk


def jax_test_cp(params, z):
    params_cosmo_power = {'Omega_cdm': [params[0]],
              'Omega_b':   [params[1]],
              'h':         [params[2]],
              'n_s':       [params[3]],
              'sigma8':    [params[4]],
             }
    params_cosmo_power['z'] = [z]

    return call_tf_full_ad(tf_test_cp,params_cosmo_power, result_shape=jnp.ones((1,540),dtype=jnp.float32))


jax_param_test = jnp.array([Omega_c_emu, Omega_b_emu, h_emu, n_s_emu, sigma8_emu])
jax_pk = jax_test_cp(jax_param_test,z_test)

jnp.allclose(jax_pk,tf_pk.numpy())

jax.jacfwd(jax_test_cp)(jax_param_test, z_test)


