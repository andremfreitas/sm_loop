import jax
import jax.numpy as jnp
from jax import random
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import time
from tqdm import tqdm
import sys

# Fixed Parameters & Useful vectors
N = 40                               # Num of shells
Ncut = 15
nu = 10**-12                           # viscosity
dt = 10**-8                           # integration step
a, b, c = (1.0, -0.50, -0.50)
k, ek, forcing = [], [], []
eps0 = 0.5 / (2**0.5)
eps1 = 0.7 * eps0
for n in range(N):
    k.append(2**n)
    ek.append(jnp.exp(-nu * dt * k[n] * k[n] / 2.0))
    if n == 0:
        forcing.append(eps0 + eps0*1j)
    if n == 1:
        forcing.append(eps1 + eps1*1j)

k_1d = jnp.array(k, dtype = jnp.complex128)
k = jnp.array(k, dtype = jnp.complex128)
ek = jnp.array(ek, dtype = jnp.complex128)
forcing = jnp.array(forcing, dtype = jnp.complex128)
forcing = jnp.concatenate((forcing, jnp.zeros(N-2)))

n_ics = 256
batch_size = n_ics
t_trasient = 20

num_steps = int(t_trasient/dt)

k = jnp.transpose(jnp.tile(k, (batch_size, 1)))
ek = jnp.transpose(jnp.tile(ek, (batch_size, 1)))
forcing = jnp.transpose(jnp.tile(forcing, (batch_size, 1)))

@jax.jit
def G(u):

    coupling = jnp.expand_dims(((a * k[0 + 1, :] * jnp.conj(u[0 + 1, :]) * u[0 + 2, :]) * 1j), axis = 0)
    coupling = jnp.concatenate([coupling, jnp.expand_dims(((a * k[1 + 1, :] * jnp.conj(u[1 + 1, :]) * u[1 + 2, :] + b * k[1, :] * jnp.conj(u[1 - 1, :]) * u[1 + 1, :]) * 1j), axis = 0)], axis = 0)

    for n in range(2, N-2):
        coupling = jnp.concatenate([coupling,
                jnp.expand_dims(((a * k[n + 1, :] * jnp.conj(u[n + 1, :]) * u[n + 2, :] + b * k[n, :] * jnp.conj(u[n - 1, :]) * u[n + 1, :] - c * k[n - 1, :] * u[n - 1, :]
                 * u[n - 2, :]) * 1j), axis=0)], axis = 0)


    coupling = jnp.concatenate([coupling, jnp.expand_dims(((b * k[N-2, :] * jnp.conj(u[N-2 - 1, :]) * u[N-2 + 1, :] - c * k[N-2 - 1, :] * u[N-2 - 1, :] * u[N-2 - 2, :]) * 1j), axis = 0)], axis = 0)

    coupling = jnp.concatenate([coupling, jnp.expand_dims(((-c * k[N-1 - 1, :] * u[N-1 - 1, :] * u[N-1 - 2, :]) * 1j), axis = 0)], axis = 0)

    return coupling

@jax.jit
def RK4(u):
    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1/2)))
    A3 = dt * (forcing + G(ek * u + A2/2))
    A4 = dt * (forcing + G(u*(ek**2) + ek*A3))

    # In terms of the original variable, the evolution rule becomes:
    u = (ek**2)*(u+A1/6) + ek*(A2+A3)/3 + A4/6
    return u

u = np.zeros((N, n_ics, 1), dtype=np.complex128)  # Store velocity vector
for i in range(n_ics):
    for n in range(6):  # Only the first 6 shells are not zero
        random = np.random.random()
        u[n, i, :] = 0.01 * k_1d[n]**(-0.33) * (np.cos(2 * np.pi * random) + 1j * np.sin(2 * np.pi * random))

aux = u[:, :, 0]

# saved_steps = 100_000
# saving_freq = 1000


num_steps = 1_000_000
start_time = time.time()

for i in tqdm(range(num_steps)):
    aux = RK4(aux)
end_time = time.time()
print('')
print(f'Time: {end_time - start_time}s')

# aux0 = aux

# aux0 = np.load('ic.npz')['aux']
# aux0 = np.load('u_final.npz')['u'][:Ncut, :, -1]


# # this second method is slightly faster but doesn't make much of a difference.
#u = jnp.zeros((N, n_ics, saved_steps+1), dtype=jnp.complex64)
# u = u.at[:, :, 0].set(jnp.complex64(aux0))
# aux = aux0
# start_time = time.time()
# for i in range(1,saved_steps+1):
#     for j in range(saving_freq):
#         aux = RK4(aux)
#     u = u.at[:, :, i].set(jnp.complex64(aux[:, :]))
#     if i % (100_000) == 0:
#         print(f'{int(i*1e-5)}s')
#         nt = np.sum(np.isnan(aux))
#         if nt != 0:
#             print('NaN somewhere ... exiting.')
#             sys.exit()
# end_time = time.time()
# print('')
# print(f'Time: {end_time - start_time}s')


# np.savez_compressed('u_40shells.npz', u = u)

