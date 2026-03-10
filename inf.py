import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# Precision
# ==========================================================

PRECISION = 32

if PRECISION == 64:
    np_c_prec = np.complex128
    tf_c_prec = tf.complex128
    tf.keras.backend.set_floatx("float64")
else:
    np_c_prec = np.complex64
    tf_c_prec = tf.complex64
    tf.keras.backend.set_floatx("float32")

# ==========================================================
# Physical parameters
# ==========================================================

N = 15
nu = 1e-12
dt = 1e-5
a, b, c = (1.0, -0.5, -0.5)

eps0 = 0.5 / np.sqrt(2)
eps1 = 0.25

batch_size = 256

# ==========================================================
# Construct shell vectors
# ==========================================================

k = []
ek = []
forcing = []

for n in range(N):
    kn = 2**n
    k.append(kn)
    ek.append(np.exp(-nu * dt * kn * kn / 2.0))
    if n == 0:
        forcing.append(eps0 + 1j * eps0)
    elif n == 1:
        forcing.append(eps1 + 1j * eps1)

k2 = [2**n for n in range(N + 2)]

k = np.array(k, dtype=np_c_prec)
ek = np.array(ek, dtype=np_c_prec)

forcing = np.array(forcing, dtype=np_c_prec)
forcing = np.concatenate((forcing, np.zeros(N - 2)))

k2_0 = np.array(k2, dtype=np_c_prec)

k = np.transpose(np.tile(k, (batch_size, 1)))
ek = np.transpose(np.tile(ek, (batch_size, 1)))
forcing = np.transpose(np.tile(forcing, (batch_size, 1)))
k2 = np.transpose(np.tile(k2_0, (batch_size, 1)))

# ==========================================================
# Load dataset
# ==========================================================

data_path = "u_40_2.npz"
model_path = "m50.keras"

data_gt = np.load(data_path)
u_gt = data_gt["u"]

n_ics = u_gt.shape[1]
num_steps = u_gt.shape[2]

print(f"Baseline data n_ics = {n_ics}")
print(f"num_steps = {num_steps}")

if n_ics % batch_size != 0:
    raise ValueError("Number of ICs must be divisible by batch_size")

u0 = u_gt[:, :, :1]
u = tf.convert_to_tensor(u0)

# ==========================================================
# Sabra operators
# ==========================================================

@tf.function(jit_compile=True)
def G(u):

    coupling = tf.expand_dims((a * k[1] * tf.math.conj(u[1]) * u[2]) * 1j, axis=0)

    coupling = tf.concat([
        coupling,
        tf.expand_dims((a * k[2] * tf.math.conj(u[2]) * u[3] +
                        b * k[1] * tf.math.conj(u[0]) * u[2]) * 1j, axis=0)
    ], axis=0)

    for n in range(2, N - 2):
        term = (a * k[n + 1] * tf.math.conj(u[n + 1]) * u[n + 2] +
                b * k[n] * tf.math.conj(u[n - 1]) * u[n + 1] -
                c * k[n - 1] * u[n - 1] * u[n - 2]) * 1j
        coupling = tf.concat([coupling, tf.expand_dims(term, axis=0)], axis=0)

    return coupling


@tf.function(jit_compile=True)
def RK4(u):

    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1/2)))
    A3 = dt * (forcing + G(ek * u + A2/2))
    A4 = dt * (forcing + G(u*(ek**2) + ek*A3))

    u = (ek**2)*(u + A1/6) + ek*(A2 + A3)/3 + A4/6

    return u

# ==========================================================
# Load trained model
# ==========================================================

model = tf.keras.models.load_model(model_path)

N1 = 3

# ==========================================================
# Time evolution
# ==========================================================

def time_evol(num_steps, u):

    for step in range(num_steps):

        aux = u[:, :, step]

        aux_real = tf.transpose(tf.math.real(aux))
        aux_im = tf.transpose(tf.math.imag(aux))

        aux_tot = tf.concat([
            tf.expand_dims(aux_real, -1),
            tf.expand_dims(aux_im, -1)
        ], axis=-1)

        pred = model(aux_tot[:, -N1:, :])

        u11 = tf.transpose(tf.complex(pred[:, 0, 0], pred[:, 0, 1]))
        u12 = tf.transpose(tf.complex(pred[:, 1, 0], pred[:, 1, 1]))

        corr_9 = dt * 1j * (a * k2[-3] * u11 * tf.math.conj(aux[-1]))
        corr_10 = dt * 1j * (a * k2[-2] * u12 * tf.math.conj(u11) +
                            b * k2[-3] * u11 * tf.math.conj(aux[-2]))

        aux = tf.convert_to_tensor(RK4(u[:, :, step]))

        aux_9 = aux[-2] + corr_9
        aux_10 = aux[-1] + corr_10

        aux_updt = tf.concat([
            aux[:N-2],
            tf.expand_dims(aux_9, 0),
            tf.expand_dims(aux_10, 0)
        ], axis=0)

        u = tf.concat([u, tf.expand_dims(aux_updt, -1)], axis=-1)

        if step % 100 == 0:
            print(step)

    return u


# ==========================================================
# Run inference
# ==========================================================

start_time = time.time()

u_pred = time_evol(num_steps, u)

end_time = time.time()

print(f"Time duration for time evolution: {end_time - start_time}")

np.savez_compressed("u_pred.npz", u=u_pred.numpy())
