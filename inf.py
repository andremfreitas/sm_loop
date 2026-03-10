import numpy as np
import tensorflow as tf
import time
import os
import sys
from tqdm import tqdm

#####################
# GENERAL STUF ######
#####################

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


np.random.seed(42)
tf.random.set_seed(42)

PRECISION = 32

if PRECISION == 64:
    np_c_prec = np.complex128
    tf_c_prec = tf.complex128
    ilayer = tf.float64
    tf.keras.backend.set_floatx('float64')
elif PRECISION == 32:
    np_c_prec = np.complex64
    tf_c_prec = tf.complex64
    ilayer = tf.float32
    tf.keras.backend.set_floatx('float32')


# Fixed Parameters & Useful vectors
N = 15  # Num of shells
nu = 10**-12  # viscosity
dt = 1 * 10**-5  # integration step
a, b, c = (1.0, -0.50, -0.50)
k, ek, forcing = [], [], []
eps0 = 0.5 / (2**0.5)
eps1 = 0.25
for n in range(N):
    k.append(2**n)
    ek.append(np.exp(-nu * dt * k[n] * k[n] / 2.0))
    if n == 0:
        forcing.append(eps0 + eps0*1j)
    if n == 1:
        forcing.append(eps1 + eps1*1j)
k2 = []
for n in range(N+2):
    k2.append(2**n)

k = np.array(k, dtype=np_c_prec)
ek = np.array(ek, dtype=np_c_prec)
forcing = np.array(forcing, dtype=np_c_prec)
forcing = np.concatenate((forcing, np.zeros(N - 2)))

batch_size = 256

k = np.transpose(np.tile(k, (batch_size, 1)))
ek = np.transpose(np.tile(ek, (batch_size, 1)))
forcing = np.transpose(np.tile(forcing, (batch_size, 1)))
k2_0 = np.array(k2, dtype=np_c_prec)
k2 = np.transpose(np.tile(k2_0, (batch_size, 1)))


jit_boolean = True         # for this small dummy data test, no XLA outperforms XLA usage. -- But for larger datasets worth it.

print("\n")

##############
# Loading data

data_path = "../../aug_train/u_40_2.npz"
data_gt = np.load(data_path)
u_gt = data_gt["u"]


print("Finished loading data")
###############

n_ics = u_gt.shape[1]
num_steps = 100_000

print(f"Baseline data n_ics = {n_ics}")
print(f"num_steps = {num_steps}")

# Preparing data
gt_tensor = tf.convert_to_tensor(u_gt, dtype=tf_c_prec)



@tf.function(jit_compile=True)
def G(u):
    
    coupling = tf.expand_dims(((a * k[0 + 1, :] * tf.math.conj(u[0 + 1, :]) * u[0 + 2, :]) * 1j), axis = 0)
    coupling = tf.concat([coupling, tf.expand_dims(((a * k[1 + 1, :] * tf.math.conj(u[1 + 1, :]) * u[1 + 2, :] + b * k[1, :] * tf.math.conj(u[1 - 1, :]) * u[1 + 1, :]) * 1j), axis = 0)], axis = 0)

    for n in range(2, N-2):
        coupling = tf.concat([coupling, 
                tf.expand_dims(((a * k[n + 1, :] * tf.math.conj(u[n + 1, :]) * u[n + 2, :] + b * k[n, :] * tf.math.conj(u[n - 1, :]) * u[n + 1, :] - c * k[n - 1, :] * u[n - 1, :] 
                * u[n - 2, :]) * 1j), axis=0)], axis = 0)


    coupling = tf.concat([coupling, tf.expand_dims(((b * k[N-2, :] * tf.math.conj(u[N-2 - 1, :]) * u[N-2 + 1, :] - c * k[N-2 - 1, :] * u[N-2 - 1, :] * u[N-2 - 2, :]) * 1j), axis = 0)], axis = 0)

    coupling = tf.concat([coupling, tf.expand_dims(((-c * k[N-1 - 1, :] * u[N-1 - 1, :] * u[N-1 - 2, :]) * 1j), axis = 0)], axis = 0)

    return coupling

@tf.function(jit_compile=True)
def RK4(u):
    A1 = dt * (forcing + G(u))        
    A2 = dt * (forcing + G(ek * (u + A1/2)))
    A3 = dt * (forcing + G(ek * u + A2/2))
    A4 = dt * (forcing + G(u*(ek**2) + ek*A3))
    
    # In terms of the original variable, the evolution rule becomes:
    u = (ek**2)*(u+A1/6) + ek*(A2+A3)/3 + A4/6
    return u


if n_ics % batch_size != 0:
    raise ValueError(
        "The number of ICs is not divisible by the specified batch size."
    )


###

u0_np = u_gt[:, :, 0:1]  # Assuming time is the third dimension
u = tf.convert_to_tensor(u0_np, dtype=tf_c_prec)

print(f"u0.shape = {u0_np.shape}")
print(f"u.shape = {u.shape}")

####################################
####################################

noise_scale = 1e-6
@tf.function(jit_compile=True)
def time_evol(model, aux, latent_dim):
    aux_real = tf.transpose(tf.math.real(aux))
    aux_im = tf.transpose(tf.math.imag(aux))
    aux_real_3d = tf.expand_dims(aux_real, axis=-1)
    aux_im_3d = tf.expand_dims(aux_im, axis=-1)
    aux_tot = tf.concat([aux_real_3d, aux_im_3d], axis=-1)  # (b, N, 2)
    
    noise = tf.random.normal((aux_tot.shape[0], latent_dim), dtype=ilayer) * noise_scale
    pred = model([aux_tot[:, -N1:, :], noise])
    
    u11 = tf.transpose(tf.complex(pred[:, 0, 0], pred[:, 0, 1]))
    u12 = tf.transpose(tf.complex(pred[:, 1, 0], pred[:, 1, 1]))

    corr_9 = dt * 1j * (a * k2[-3] * u11 * tf.math.conj(aux[-1, :]))
    corr_10 = dt * 1j * (a * k2[-2] * u12 * tf.math.conj(u11) + b * k2[-3] * u11 * tf.math.conj(aux[-2, :]))

    aux_new = tf.convert_to_tensor(RK4(aux))
    aux_9 = aux_new[-2, :] + corr_9
    aux_10 = aux_new[-1, :] + corr_10

    aux_updt = tf.concat([aux_new[:N - 2, :], tf.expand_dims(aux_9, axis=0), tf.expand_dims(aux_10, axis=0)], axis=0)
    return aux_updt



directories = ['m15_1e-6_ld1/']

for directory in directories:
    keras_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.keras'):
            keras_files.append(filename)

    for model_name in keras_files:
        # Loading model
        path_to_dir = directory + model_name
        model = tf.keras.models.load_model(path_to_dir)           
        
        N1 = 3           # must match training
        latent_dim = 1   # must match training


        u_storage = tf.Variable(tf.zeros([N, batch_size, num_steps], dtype=tf_c_prec))

        ib = 0
        batch_start = ib
        batch_end = ib + batch_size
        u_mp = u[:, batch_start:batch_end, -1]
        start_time = time.time()
        for i in tqdm(range(num_steps)):
            u_mp = time_evol(model, u_mp, latent_dim)
            u_storage[:, :, i].assign(u_mp)
        end_time = time.time()

        print(f"Time duration for time evolution: {end_time - start_time}")

        np.savez_compressed(directory + "u_" + model_name + ".npz", u = u_storage.numpy())
        # np.savez_compressed(directory + "u_" + model_name + ".npz", u = u)

