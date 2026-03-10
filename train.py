import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import sys

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
    ilayer = tf.float64
    tf.keras.backend.set_floatx("float64")
else:
    np_c_prec = np.complex64
    tf_c_prec = tf.complex64
    ilayer = tf.float32

# ==========================================================
# Physical parameters
# ==========================================================

N = 15
nu = 1e-12
dt = 1e-5
a, b, c = (1.0, -0.5, -0.5)

eps0 = 0.5 / np.sqrt(2)
eps1 = 0.25

batch_size = 1024

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

k0 = np.array(k, dtype=np_c_prec)
ek0 = np.array(ek, dtype=np_c_prec)

forcing = np.array(forcing, dtype=np_c_prec)
forcing0 = np.concatenate((forcing, np.zeros(N - 2, dtype=np_c_prec)))

k2_0 = np.array(k2, dtype=np_c_prec)

# ==========================================================
# Neural network closure
# ==========================================================

def corrector(N1, N2, num_layers, hidden_size, batch_size=None):
    inputs = keras.Input(shape=(N1, 2), batch_size=batch_size)
    x = keras.layers.Flatten()(inputs)
    for _ in range(num_layers):
        x = keras.layers.Dense(hidden_size, activation="relu")(x)
    x = keras.layers.Dense(N2 * 2)(x)
    outputs = keras.layers.Reshape((N2, 2))(x)
    return keras.Model(inputs=inputs, outputs=outputs)

N1 = 3
N2 = 2
num_layers = 7
hidden_size = 256

model = corrector(N1, N2, num_layers, hidden_size, batch_size)
model.summary()

variables = model.trainable_variables

# ==========================================================
# Optimizer
# ==========================================================

lr0 = 3e-4
optimizer = keras.optimizers.Adam(lr0)
model.compile(optimizer=optimizer)

# ==========================================================
# Load training data
# ==========================================================

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "../../aug_train/u_40_2.npz"

print("Loading dataset:", data_path)

data_gt = np.load(data_path)
u_gt = data_gt["u"]

print("Dataset shape:", u_gt.shape)

n_ics0 = u_gt.shape[1]
num_steps0 = u_gt.shape[2]

print(f"Baseline data n_ics = {n_ics0}")
print(f"num_steps = {num_steps0}")

# ==========================================================
# Sabra model operators
# ==========================================================

jit_boolean = True

@tf.function(jit_compile=jit_boolean)
def G(u):
    coupling = tf.expand_dims((a * k[1] * tf.math.conj(u[1]) * u[2]) * 1j, axis=0)
    coupling = tf.concat([coupling,
        tf.expand_dims((a * k[2] * tf.math.conj(u[2]) * u[3] + b * k[1] * tf.math.conj(u[0]) * u[2]) * 1j, axis=0)], axis=0)
    for n in range(2, N - 2):
        term = (a * k[n + 1] * tf.math.conj(u[n + 1]) * u[n + 2]
                + b * k[n] * tf.math.conj(u[n - 1]) * u[n + 1]
                - c * k[n - 1] * u[n - 1] * u[n - 2]) * 1j
        coupling = tf.concat([coupling, tf.expand_dims(term, axis=0)], axis=0)
    return coupling

@tf.function(jit_compile=True)
def RK4(u):
    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1 / 2)))
    A3 = dt * (forcing + G(ek * u + A2 / 2))
    A4 = dt * (forcing + G(u * (ek**2) + ek * A3))
    u = (ek**2) * (u + A1 / 6) + ek * (A2 + A3) / 3 + A4 / 6
    return u

# ==========================================================
# Training step
# ==========================================================

@tf.function(jit_compile=False)
def training_loop(u0, gt_tensor, msteps):
    with tf.GradientTape() as tape:
        u = u0
        for i in range(msteps - 1):
            aux = u[:, :, i]
            aux_real = tf.transpose(tf.math.real(aux))
            aux_im = tf.transpose(tf.math.imag(aux))
            aux_tot = tf.concat([tf.expand_dims(aux_real, -1), tf.expand_dims(aux_im, -1)], axis=-1)

            pred = model(aux_tot[:, -N1:, :])

            u11 = tf.transpose(tf.complex(pred[:, 0, 0], pred[:, 0, 1]))
            u12 = tf.transpose(tf.complex(pred[:, 1, 0], pred[:, 1, 1]))

            corr_9 = dt * 1j * (a * k2[-3] * u11 * tf.math.conj(aux[-1]))
            corr_10 = dt * 1j * (a * k2[-2] * u12 * tf.math.conj(u11) + b * k2[-3] * u11 * tf.math.conj(aux[-2]))

            aux = tf.convert_to_tensor(RK4(u[:, :, i]))

            aux_9 = aux[-2] + corr_9
            aux_10 = aux[-1] + corr_10

            aux_updt = tf.concat([aux[:N - 2],
                                  tf.expand_dims(aux_9, 0),
                                  tf.expand_dims(aux_10, 0)], axis=0)

            u = tf.concat([u, tf.expand_dims(aux_updt, -1)], axis=-1)

        loss = tf.reduce_mean(tf.reduce_mean(
            tf.square(tf.abs(u[-6:] - gt_tensor[-6:])),
            axis=(1, 2)))

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

# ==========================================================
# Training schedule
# ==========================================================

msteps_interval = [2, 4, 10, 15, 20, 25, 30, 40, 50]
msteps_chosen = 50

msteps_sched = []
epochs = []

for msteps in msteps_interval:
    msteps_sched.append(msteps)
    if msteps == msteps_chosen:
        epochs.append(50)
        break
    epochs.append(1)

print("Training schedule:", msteps_sched)
print("Epoch schedule:", epochs)

# ==========================================================
# Utilities
# ==========================================================

def plot_loss(losses, msteps):
    plt.plot(losses, linewidth=2)
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"loss_{msteps}.png")
    plt.close()

def closest_divisible(x, y):
    return x - (x % y)

# ==========================================================
# Training loop
# ==========================================================

start = time.time()
batch_losses = []

for j in range(len(epochs)):
    msteps = msteps_sched[j]
    print(f"\nTraining with msteps = {msteps}\n")

    num_steps_new = closest_divisible(num_steps0, msteps)

    gt_reshaped = tf.reshape(
        u_gt[:, :, :num_steps_new],
        [N, int(n_ics0 * num_steps_new / msteps), msteps])

    gt_reshaped = tf.transpose(gt_reshaped, (1, 0, 2))

    dataset = tf.data.Dataset.from_tensor_slices(gt_reshaped)
    dataset = dataset.batch(batch_size)

    losses = []

    for epoch in range(epochs[j]):
        dataset_shuffled = dataset.shuffle(buffer_size=batch_size)

        for gt in dataset_shuffled:
            gt = tf.transpose(gt, (1, 0, 2))
            ic = gt[:N, :, :1]

            loss = training_loop(ic, gt, msteps)

            batch_losses.append(loss.numpy())
            if np.isnan(loss.numpy()):
                print("NaN detected — stopping training.")
                sys.exit(1)

        epoch_loss = np.mean(batch_losses[-batch_size:])
        losses.append(epoch_loss)
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

plot_loss(losses, msteps)

tf.keras.models.save_model(model, f"m{msteps}.keras")

end = time.time()
print(f"Training duration: {end-start:.2f}s")
