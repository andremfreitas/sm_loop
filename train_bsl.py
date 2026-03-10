import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import time
import sys
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

k0 = np.array(k, dtype=np_c_prec)
ek0 = np.array(ek, dtype=np_c_prec)
forcing = np.array(forcing, dtype=np_c_prec)
forcing0 = np.concatenate((forcing, np.zeros(N - 2, dtype = np_c_prec)))

#n_ics = 100
batch_size = 1024

k = np.transpose(np.tile(k0, (batch_size, 1)))
ek = np.transpose(np.tile(ek0, (batch_size, 1)))
forcing = np.transpose(np.tile(forcing0, (batch_size, 1)))
k2_0 = np.array(k2, dtype=np_c_prec)
k2 = np.transpose(np.tile(k2_0, (batch_size, 1)))

def corrector(N1, N2, num_layers, hidden_size, batch_size=None):
    # Define the input layer with shape (N1, 2)
    inputs = keras.Input(shape=(N1, 2), batch_size=batch_size)
    
    # Flatten the input from (N1, 2) to (N1 * 2)
    x = keras.layers.Flatten()(inputs)
    
    # Pass through the dense layers
    for _ in range(num_layers):
        x = keras.layers.Dense(units=hidden_size, activation="relu")(x)
    
    # Final dense layer to produce (N2 * 2) outputs
    x = keras.layers.Dense(units=N2 * 2, activation="linear")(x)
    
    # Reshape the output to (N2, 2)
    outputs = keras.layers.Reshape((N2, 2))(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Example usage
N1 = 3  # Number of input features
N2 = 2  # Number of output features
num_layers = 7  # Number of hidden layers
hidden_size = 256  # Number of units in each hidden layer

model = corrector(N1, N2, num_layers, hidden_size, batch_size)
model.summary()
variables = model.trainable_variables

# lr0 = 1e-3
lr0 = 3e-4
optimizer = keras.optimizers.Adam(lr0)
model.compile(optimizer=optimizer)
epochs = 10


jit_boolean = True        # for this small dummy data test, no XLA outperforms XLA usage. -- But for larger datasets worth it.

print("\n")

##############
# Loading data

data_path = "../../aug_train/u_40_2.npz"
# data_path = "../jjax/u_final.npz"
data_gt = np.load(data_path)
u_gt = data_gt["u"]         
print(u_gt.shape)


print("Finished loading data")
###############

n_ics0 = u_gt.shape[1]
num_steps0 = u_gt.shape[2]

print(f"Baseline data n_ics = {n_ics0}")
print(f"num_steps = {num_steps0}")




@tf.function(jit_compile=jit_boolean)
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



@tf.function(jit_compile=False)
def training_loop(u0, gt_tensor, msteps):
    with tf.GradientTape() as tape:
        u = u0
        for i in range(msteps-1):
            aux = u[:, :, i]
            aux_real = tf.transpose(tf.math.real(aux))
            aux_im = tf.transpose(tf.math.imag(aux))
            aux_real_3d = tf.expand_dims(aux_real, axis=-1)
            aux_im_3d = tf.expand_dims(aux_im, axis=-1)
            aux_tot = tf.concat([aux_real_3d, aux_im_3d], axis=-1)  # (b, N, 2)
            pred = model(aux_tot[:, -N1:, :])
            u11 = tf.transpose(tf.complex(pred[:, 0, 0], pred[:, 0, 1]))
            u12 = tf.transpose(tf.complex(pred[:, 1, 0], pred[:, 1, 1]))

            corr_9 = dt * 1j * (a * k2[-3] * u11 * tf.math.conj(aux[-1, :]))
            corr_10 = dt * 1j * (a * k2[-2] * u12 * tf.math.conj(u11) + b * k2[-3] * u11 * tf.math.conj(aux[-2, :]))

            aux = tf.convert_to_tensor(RK4(u[:, :, i]))
            aux_9 = aux[-2, :] + corr_9
            aux_10 = aux[-1, :] + corr_10

            aux_updt = tf.concat([aux[:N-2, :], tf.expand_dims(aux_9, axis = 0), tf.expand_dims(aux_10, axis = 0)], axis = 0)

            aux_updt_3d = tf.expand_dims(aux_updt, axis=-1)
            u = tf.concat([u, aux_updt_3d], axis=-1)

        loss = tf.reduce_mean(tf.reduce_mean((tf.square(tf.math.abs(u[-6:, :, :] - gt_tensor[-6:, :, :]))), axis = (1,2)) /\
         ((tf.sqrt(tf.reduce_mean(tf.square(tf.abs(gt_tensor[-6:, :, :])), axis = (1,2)))) * tf.sqrt(tf.reduce_mean(tf.square(tf.abs(u[-6:, :, :])), axis = (1,2)))))
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

losses = []
batchwise_losses = []

# msteps_interval = [2, 4, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
msteps_interval = [2, 4, 10, 15, 20, 25, 30, 40, 50]


# msteps_chosen = int(sys.argv[1])
msteps_chosen = 50

msteps_sched = []
epochs = []

for msteps in msteps_interval:
    msteps_sched.append(msteps)
    if msteps == msteps_chosen:
        epochs.append(50)
        break
    epochs.append(1)

print(msteps_sched)
print(epochs)

def lr_schedule(lr0, factor, last_epochs):
    lr = [lr0]
    for _ in range(last_epochs):
        lr_ip1 = lr[-1] * factor
        lr.append(lr_ip1)
    return lr

lr = [8e-4, 6e-4, 4e-4, 3e-4]
#lr = [3e-4, 1e-4, 1e-4, 1e-4]

# factor = 0.91
factor = 1.0
last_epochs = epochs[-1]
lr_20 = lr_schedule(lr[-1], factor, last_epochs)

counter_20 = 0
counter = 0

# ############
# #  PLOTTING
# ############


def plot_loss(losses, msteps):
    plt.plot(losses, linewidth = 2)
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"loss_{msteps}.png")
    plt.close()


def closest_divisible(x, y):
    # Find the closest value below or equal to x that is divisible by y
    return x - (x % y)


start = time.time()
for j in range(len(epochs)):
    msteps = msteps_sched[j]
    print(f"\n msteps = {msteps} \n")

    num_steps_new = closest_divisible(num_steps0, msteps)

    #print(f'u_gt[:, :, :num_steps_new].shape = {u_gt[:, :, :num_steps_new].shape}')

    gt_reshaped = tf.reshape(u_gt[:, :, :num_steps_new], [N, int(n_ics0 * num_steps_new / msteps), msteps]) 

    n_ics = gt_reshaped.shape[1]

    #print(f'n_ics = {n_ics}')

    gt_reshaped = tf.transpose(gt_reshaped, (1,0,2))

    data = tf.data.Dataset.from_tensor_slices(gt_reshaped)
    bdata = data.batch(batch_size = batch_size)
    losses = []
    for epoch in range(epochs[j]):

        sbdata = bdata.shuffle(buffer_size = batch_size)

        for gt in sbdata:
            gt = tf.transpose(gt, (1,0,2))
            ic = gt[:N, :, :1]
            bs = gt.shape[1]
            k = np.transpose(np.tile(k0, (bs, 1)))
            ek = np.transpose(np.tile(ek0, (bs, 1)))
            forcing = np.transpose(np.tile(forcing0, (bs, 1)))
            k2 = np.transpose(np.tile(k2_0, (bs, 1)))
            loss = training_loop(ic, gt, msteps_sched[j])    
            batchwise_losses.append(loss.numpy()) 
            if np.isnan(loss.numpy()):
                print('NaN found in loss. Stopping the run ...')
                sys.exit(1)   
        
        losses.append(sum(batchwise_losses[-batch_size:]) / batch_size)
        print(f"Epoch {epoch}, Loss: {sum(batchwise_losses[-batch_size:]) / batch_size}")

        # if j == (len(epochs) - 1):
        #     optimizer.learning_rate.assign(lr_20[counter_20])
        #     counter_20 += 1
        #     print(f'lr = {model.optimizer.learning_rate.numpy()}')

        # if epoch == (epochs[j] - 1) and j != (len(epochs) - 1):
        #     optimizer.learning_rate.assign(lr[counter])
        #     counter += 1
        #     print(f'lr = {model.optimizer.learning_rate.numpy()}')

        # if epoch == (epochs[j] - 1):
        #     print('Checkpoint: saving model')
        #     tf.keras.models.save_model(model, f"msteps_test/m{msteps}.keras")

plot_loss(losses, msteps)

tf.keras.models.save_model(model, f"m{msteps}.keras")

end = time.time()
print(f"Training duration:{end-start}s")


