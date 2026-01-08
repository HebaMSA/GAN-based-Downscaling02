
import numpy as np
import netCDF4 as nc
import tensorflow as tf
import os
import time
import pandas as pd

# Set GPU memory growth before initializing strategy or using any GPU resources
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: ", strategy.num_replicas_in_sync)

# Now import Keras components
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Ensure deterministic execution
# tf.config.experimental.enable_op_determinism()
# tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Print TensorFlow version and GPU availability
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# .............................................
# PROCESSING NETCDF FILES
# .............................................

def process_file(file_path):
    with nc.Dataset(file_path, mode='r') as nc_file:
        available_vars = nc_file.variables.keys()
        var_name = 'PREC' if 'PREC' in available_vars else 'pr' if 'pr' in available_vars else None
        if not var_name:
            raise ValueError(f"Unknown variable in {file_path}. Available variables: {available_vars}")
        return nc_file.variables[var_name][:]

def process_directory(directory_path):
    all_data = []
    spatial_shape = None
    
    for file_name in sorted(os.listdir(directory_path)):
        if file_name.endswith('.nc'):
            file_path = os.path.join(directory_path, file_name)
            data = process_file(file_path)
            if spatial_shape is None:
                spatial_shape = data.shape[1:]
            elif spatial_shape != data.shape[1:]:
                raise ValueError(f"Spatial dimensions mismatch! Expected {spatial_shape}, but got {data.shape[1:]} in {file_name}")
            all_data.append(data)
    
    full_data = np.concatenate(all_data, axis=0)
    num_days = full_data.shape[0] // 24
    reshaped_data = full_data[:num_days * 24].reshape(num_days, 24, *spatial_shape)
    return np.expand_dims(reshaped_data, axis=-1)

# Load and preprocess data
y_train = process_directory('/lustre06/project/6090487/hma153/Mitacs/WRF_Mitacs/orig_adj2/pgw/')
x_train = process_directory('/lustre06/project/6090487/hma153/Mitacs/WRF_Mitacs/up_adj2/pgw/')

total_samples = x_train.shape[0]
train_end = int(total_samples * 0.8)
val_end = train_end + int(total_samples * 0.1)

Xtrain, Ytrain = x_train[:train_end], y_train[:train_end]
Xval, Yval = x_train[train_end:val_end], y_train[train_end:val_end]
Xtest, Ytest = x_train[val_end:], y_train[val_end:]

print(np.min(Ytrain))
print(np.max(Ytrain))

print(np.min(Xtrain))
print(np.max(Xtrain))

print(np.min(Yval))
print(np.max(Yval))

print(np.min(Xval))
print(np.max(Xval))

print(np.min(Ytest))
print(np.max(Ytest))

print(np.min(Xtest))
print(np.max(Xtest))


print("Xtrain type:", type(Xtrain))
print("Ytrain type:", type(Ytrain))
print("Xtrain dtype:", getattr(Xtrain, 'dtype', 'N/A'))
print("Ytrain dtype:", getattr(Ytrain, 'dtype', 'N/A'))
print("Xtrain shape:", getattr(Xtrain, 'shape', 'N/A'))
print("Ytrain shape:", getattr(Ytrain, 'shape', 'N/A'))

Xtrain = np.array(Xtrain, dtype=np.float32)
Ytrain = np.array(Ytrain, dtype=np.float32)

print("Converted shapes:", Xtrain.shape, Ytrain.shape)


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

# Residual Block function
def residual_block(x, filters, kernel_size=3, stride=1, padding='same', DROP_RATE = 0.2, l2_reg=0.0005):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding, 
                      kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(DROP_RATE)(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding, 
                      kernel_initializer="he_normal",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding=padding, 
                                 kernel_initializer="he_normal",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

#The thresholding layer
class ThresholdLayer(Layer):
    def call(self, inputs):
        return tf.where(inputs < 0, tf.zeros_like(inputs), inputs)

from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * tf.constant(0.5, dtype=tf.float32)

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

class GeneratorLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, optimizer, schedule_fn):
        super().__init__()
        self.optimizer = optimizer
        self.schedule_fn = schedule_fn

    def on_epoch_begin(self, epoch, logs=None):
        old_lr = float(tf.keras.backend.get_value(self.optimizer.lr))
        new_lr = self.schedule_fn(epoch, old_lr)
        tf.keras.backend.set_value(self.optimizer.lr, new_lr)
        print(f"\nGenerator LR adjusted: {old_lr:.6f} ? {new_lr:.6f}")

# Generator Model
# Define the generator model
def make_generator(input_shape=(24, 26, 42, 1), noise_shape=(26, 42, 1), output_shape=(24, 364, 588, 1), DROP_RATE=0.2, l2_reg=0.0005):
    low_res_input = layers.Input(shape=input_shape, name='low_res_input')
    noise_input = layers.Input(shape=noise_shape, name='noise_input')

    # Noise Processing (Fewer Parameters)
    noise_embedding = layers.Conv2D(8, kernel_size=1, activation='relu', kernel_initializer="he_normal")(noise_input)
    shared_noise_repeated = tf.repeat(tf.expand_dims(noise_embedding, axis=1), repeats=input_shape[0], axis=1)

    x_init = layers.TimeDistributed(layers.Conv2D(8, kernel_size=3, padding='same', kernel_initializer="he_normal"))(low_res_input)
    x_init = layers.LeakyReLU(alpha=0.2)(x_init)
    x_init = layers.Concatenate(axis=-1)([x_init[:, 0], noise_input])
    x_init = residual_block(x_init, filters=8)

    # Reduced ConvLSTM2D Filters + Dropout
    init_state = layers.Conv2D(8, kernel_size=3, padding='same', activation='relu', kernel_initializer="he_normal")(x_init)

    # Transform init_state to match ConvLSTM2D filters
    init_state_transformed = layers.Conv2D(16, kernel_size=1, padding='same', activation='relu', kernel_initializer="he_normal")(init_state)

    x_update = []
    for t in range(output_shape[0]):
        x_t = layers.TimeDistributed(layers.Conv2D(8, kernel_size=3, padding='same', kernel_initializer="he_normal"))(low_res_input)[:, t]
        x_t = layers.LeakyReLU(alpha=0.2)(x_t)
        x_t = layers.Dropout(DROP_RATE)(x_t)  # Apply dropout
        x_t = layers.Concatenate(axis=-1)([x_t, shared_noise_repeated[:, t]])
        x_t = residual_block(x_t, filters=8)
        x_update.append(x_t)

    x_update = tf.stack(x_update, axis=1)
    x_update = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu', dropout=DROP_RATE)(x_update, initial_state=[init_state_transformed, init_state_transformed])

    # First upsampling block (26x42 ? 52x84)
    x_up = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)))(x_update)
    x_up = layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)

    # Second upsampling block (52x84 ? 104x168)
    x_up = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)))(x_up)
    x_up = layers.TimeDistributed(layers.Conv2D(16, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)

    # Third upsampling block (104x168 ? 208x336)
    x_up = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)))(x_up)
    x_up = layers.TimeDistributed(layers.Conv2D(8, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)

    # Fourth upsampling block (208x336 ? 416x672)
    x_up = layers.TimeDistributed(layers.UpSampling2D(size=(2, 2)))(x_up)
    x_up = layers.TimeDistributed(layers.Conv2D(4, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)

    # Final resize to match exact dimensions (364, 588)
    x_up = layers.TimeDistributed(layers.Resizing(364, 588, interpolation='bilinear'))(x_up)
    
    # Final convolution layers
    x_up = layers.TimeDistributed(layers.Conv2D(8, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)
    x_up = layers.TimeDistributed(layers.Conv2D(4, kernel_size=3, padding='same'))(x_up)
    x_up = layers.TimeDistributed(layers.LeakyReLU(0.2))(x_up)
    x_up = layers.TimeDistributed(layers.Conv2D(1, kernel_size=3, padding='same', activation='linear'))(x_up)

    # Applying the thresholding layer
    output = layers.TimeDistributed(ThresholdLayer())(x_up)

    model = Model(inputs=[low_res_input, noise_input], outputs=output, name='generator')

    return model



import tensorflow as tf
from tensorflow.keras import layers, Model

def leaky_relu_with_slope(x):
    return layers.LeakyReLU(alpha=0.2)(x)

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(f'SpectralNormalization must wrap a `Layer` instance. You passed: {layer}')
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False, name='sn_v', dtype=tf.float32)
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False, name='sn_u', dtype=tf.float32)
        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        v_hat = self.v

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.norm(v_) + self.eps)
                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.norm(u_) + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)
        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)

def time_distributed_residual_block(x, filters, kernel_size=3, stride=1, padding='same'):
    shortcut = x

    x = layers.TimeDistributed(
        SpectralNormalization(
            layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding)
        )
    )(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)

    x = layers.TimeDistributed(
        SpectralNormalization(
            layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding=padding)
        )
    )(x)

    if shortcut.shape[-1] != filters or shortcut.shape[2] != x.shape[2] or shortcut.shape[3] != x.shape[3]:
        shortcut = layers.TimeDistributed(
            SpectralNormalization(
                layers.Conv2D(filters, kernel_size=1, strides=stride, padding=padding)
            )
        )(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    return x

class CustomBilinearResize(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width):
        super(CustomBilinearResize, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return self._resize_with_custom_grad(inputs)

    def _resize_with_custom_grad(self, inputs):
        @tf.custom_gradient
        def resize_op(x):
            resized = tf.image.resize(x, [self.target_height, self.target_width], method='bilinear')

            def grad(dy):
                # Stop gradient as fallback
                return tf.image.resize(dy, tf.shape(x)[1:3], method='bilinear')

            return resized, grad

        return resize_op(inputs)
def make_discriminator():
    input_highres = tf.keras.Input(shape=(24, 364, 588, 1))
    input_lowres = tf.keras.Input(shape=(24, 26, 42, 1))

    # High-res path (reduced filters)
    x_high = time_distributed_residual_block(input_highres, filters=8, stride=2)   # was 12
    x_high = time_distributed_residual_block(x_high, filters=12, stride=2)         # was 16
    x_high = time_distributed_residual_block(x_high, filters=16, stride=2)         # was 20
    x_high = time_distributed_residual_block(x_high, filters=16, stride=2)         # was 20

    # Low-res path (reduced filters)
    x_low = time_distributed_residual_block(input_lowres, filters=8, stride=2)     # was 12
    x_low = time_distributed_residual_block(x_low, filters=12, stride=2)           # was 16
    x_low = time_distributed_residual_block(x_low, filters=16, stride=2)           # was 20
    x_low = time_distributed_residual_block(x_low, filters=16, stride=1)           # was 20

    # Match spatial dims
    x_low = layers.TimeDistributed(CustomBilinearResize(24, 36))(x_low)
    x_high = layers.TimeDistributed(CustomBilinearResize(24, 36))(x_high)

    # Concatenate + residual
    concatenated = layers.Concatenate(axis=-1)([x_high, x_low])
    concatenated = time_distributed_residual_block(concatenated, filters=20, stride=1)  # was 24

    # Conv block (reduced filters)
    x_combined = layers.TimeDistributed(
        SpectralNormalization(
            layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')  # was 24
        )
    )(concatenated)
    x_combined = layers.GlobalAveragePooling3D()(x_combined)

    # High-res only stream (reduced filters)
    x_high_processed = layers.TimeDistributed(
        SpectralNormalization(
            layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')  # was 24
        )
    )(x_high)
    x_high_processed = layers.GlobalAveragePooling3D()(x_high_processed)

    # Final MLP (reduced units)
    final_concat = layers.Concatenate()([x_combined, x_high_processed])

    x = layers.Dense(24,
                     use_bias=True,
                     kernel_regularizer=tf.keras.regularizers.l1_l2(0.0, 1e-4),
                     kernel_initializer=tf.keras.initializers.HeNormal(),
                     bias_initializer=tf.keras.initializers.RandomNormal(seed=8889))(final_concat)
    x = leaky_relu_with_slope(x)

    x = layers.Dense(12,
                     use_bias=True,
                     kernel_regularizer=tf.keras.regularizers.l1_l2(0.0, 1e-4),
                     kernel_initializer=tf.keras.initializers.HeNormal(),
                     bias_initializer=tf.keras.initializers.RandomNormal(seed=8889))(x)
    x = leaky_relu_with_slope(x)

    x = layers.Dense(6,
                     use_bias=True,
                     kernel_regularizer=tf.keras.regularizers.l1_l2(0.0, 1e-4),
                     kernel_initializer=tf.keras.initializers.HeNormal(),
                     bias_initializer=tf.keras.initializers.RandomNormal(seed=8889))(x)
    x = leaky_relu_with_slope(x)

    output = layers.Dense(1, activation='linear',
                          use_bias=True,
                          kernel_regularizer=tf.keras.regularizers.l1_l2(0.0, 1e-4),
                          kernel_initializer=tf.keras.initializers.HeNormal(),
                          bias_initializer=tf.keras.initializers.RandomNormal(seed=8889))(x)

    model = Model(inputs=[input_highres, input_lowres], outputs=output)
    model.summary()
    return model

# Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import csv

# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: ", strategy.num_replicas_in_sync)

# WGAN Class Definition
class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=1.0):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.val_d_loss_tracker = keras.metrics.Mean(name="val_d_loss")
        self.val_g_loss_tracker = keras.metrics.Mean(name="val_g_loss")


    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def test_step(self, data):
        low_res, high_res = data
        batch_size = tf.shape(low_res)[0]

        noise = tf.random.normal([batch_size] + list(self.latent_dim[1:]), mean=0.0, stddev=0.01)
        fake = self.generator([low_res, noise], training=False)

        fake_logits = self.discriminator([fake, low_res], training=False)
        real_logits = self.discriminator([high_res, low_res], training=False)

        d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
        g_loss = self.g_loss_fn(fake_logits)

        # Track validation losses in custom metrics
        self.val_d_loss_tracker.update_state(d_loss)
        self.val_g_loss_tracker.update_state(g_loss)

        # Required return format for Keras
        return {
        "val_d_loss": d_loss,
        "val_g_loss": g_loss
        }

    def reset_metrics(self):
        super().reset_metrics()
        self.val_d_loss_tracker.reset_states()
        self.val_g_loss_tracker.reset_states()
    
    @property
    def metrics(self):
        return [self.val_d_loss_tracker, self.val_g_loss_tracker]

    def gradient_penalty(self, batch_size, low_res, real, fake):
        alpha = tf.random.uniform((batch_size, 1, 1, 1, 1), 0.0, 1.0)
        interpolated = real + alpha * (fake - real)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, low_res], training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1e-10)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        low_res, high_res = data
        batch_size = tf.shape(low_res)[0]

        for _ in range(self.d_steps):
            noise = tf.random.normal([batch_size] + list(self.latent_dim[1:]), mean=0.0, stddev=0.01)
            with tf.GradientTape() as tape:
                fake = self.generator([low_res, noise], training=True)
                fake_logits = self.discriminator([fake, low_res], training=True)
                real_logits = self.discriminator([high_res, low_res], training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, low_res, high_res, fake)
                d_loss = d_cost + self.gp_weight * gp
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size] + list(self.latent_dim[1:]), mean=0.0, stddev=0.01)
        with tf.GradientTape() as tape:
            fake = self.generator([low_res, noise], training=True)
            gen_logits = self.discriminator([fake, low_res], training=True)
            g_loss = self.g_loss_fn(gen_logits)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss, "gradient_penalty": gp}

# Custom CSV Logger
class CSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename='loss_log.csv', generator_optimizer=None):
        super().__init__()
        self.filename = filename
        self.generator_optimizer = generator_optimizer  # <-- pass this from outside
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "epoch", "d_loss", "g_loss", "gradient_penalty",
                    "val_d_loss", "val_g_loss", "generator_lr"
                ])

    def on_epoch_end(self, epoch, logs=None):
        # Get generator learning rate
        lr = tf.keras.backend.get_value(self.generator_optimizer.lr) if self.generator_optimizer else None

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                logs.get("d_loss"),
                logs.get("g_loss"),
                logs.get("gradient_penalty"),
                logs.get("val_d_loss"),
                logs.get("val_g_loss"),
                lr
            ])

# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: ", strategy.num_replicas_in_sync)

BATCH_SIZE = 32
latent_dim = (BATCH_SIZE, 26, 42, 1)

# Build models inside strategy scope
with strategy.scope():
    generator = make_generator()
    generator.load_weights("generator_df_modelV11_pgw_weights.h5")
    print("? Weights successfully loaded!")
    generator.summary()

    discriminator = make_discriminator()

    generator_optimizer = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.5, beta_2=0.9)

    def discriminator_loss(real_img, fake_img):
        return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    wgan = WGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0
    )

    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss
    )

csv_logger = CSVLogger(filename='wgan_loss_ph02_V16t_new01p.csv', generator_optimizer=generator_optimizer)

# Dataset generator
def data_gen():
    for x, y in zip(Xtrain, Ytrain):
        yield x, y

output_signature = (
    tf.TensorSpec(shape=Xtrain.shape[1:], dtype=tf.float32),
    tf.TensorSpec(shape=Ytrain.shape[1:], dtype=tf.float32)
)

batches = tf.data.Dataset.from_generator(data_gen, output_signature=output_signature).batch(BATCH_SIZE)

def val_data_gen():
    for x, y in zip(Xval, Yval):
        yield x, y

val_batches = tf.data.Dataset.from_generator(val_data_gen, output_signature=output_signature).batch(BATCH_SIZE)

gen_lr_callback = GeneratorLRScheduler(optimizer=generator_optimizer, schedule_fn=scheduler)

# Train
wgan.fit(batches, epochs=150, shuffle=True, validation_data=val_batches, callbacks=[csv_logger, gen_lr_callback])

# Save the model

generator.save("WGAN_phase02_V16t_new01p.keras")

# Save the trained model outside the function
generator.save("WGAN_phase02_V16t_new01p.h5")
generator.save_weights("WGAN_phase02_V16t_new01p_weights.h5")

print("Model saved as WGAN_phase02_V16t_new01p.keras")
