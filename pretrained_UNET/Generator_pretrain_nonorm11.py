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
y_train = process_directory('/lustre06/project/6090487/hma153/Mitacs/WRF_Mitacs/orig_adj2/ctl/')
x_train = process_directory('/lustre06/project/6090487/hma153/Mitacs/WRF_Mitacs/up_adj2/ctl/')

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

from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * tf.constant(0.5, dtype=tf.float32)

lr_scheduler = LearningRateScheduler(scheduler)

# Callbacks
class PrintEpochLoss(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")

class CSVLoggerCallback(Callback):
    def __init__(self, filename="training_log.csv"):
        super().__init__()
        self.filename = filename
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append({"epoch": epoch + 1, "loss": logs["loss"], "val_loss": logs["val_loss"]})

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.filename, index=False)
        print(f"Training and validation losses saved to {self.filename}")



# .............................................
# GENERATOR MODEL
# .............................................
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

def generator_fit(train_dataset, x_val, y_val,
                  input_shape=(24, 26, 42, 1), noise_shape=(26, 42, 1), 
                  output_shape=(24, 364, 588, 1), BATCH_SIZE=2, LEARNING_RATE=0.00001, DROP_RATE=0.2, 
                  NUM_EPOCHS=5, PATIENCE=3, VERBOSITY=1, l2_reg=0.0005, log_filename="training_log_df_V11.csv"):
    """
    Function to create, train, and evaluate a generator model that upscales (24, 26, 42, 1) to (24, 364, 588, 1),
    now using tf.data.Dataset for training input.
    """

    # Residual Block function
    def residual_block(x, filters, kernel_size=3, stride=1, padding='same', l2_reg=0.0005):
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

    # Generator Model
    # Define the generator model
    def make_generator(input_shape, noise_shape, output_shape):
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

    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE, clipvalue=0.5)
    generator = make_generator(input_shape, noise_shape, output_shape)
    generator.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=VERBOSITY)
    csv_logger = CSVLoggerCallback(log_filename)
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model
    history = generator.fit(
        x=train_dataset,
        validation_data=(
            [x_val, np.random.normal(0, 1, (x_val.shape[0],) + noise_shape)],
            y_val
        ),
        epochs=NUM_EPOCHS,
        shuffle=True,
        verbose=VERBOSITY,
        callbacks=[early_stopping, csv_logger, lr_scheduler]
    )

    return generator, history


# Define noise_shape before its first use
noise_shape = (26, 42, 1)  # Define noise shape to match the spatial dimensions of input

def data_gen():
    for x, y in zip(Xtrain, Ytrain):
        noise = np.random.normal(0, 1, size=noise_shape).astype(np.float32)
        yield (x.astype(np.float32), noise), y.astype(np.float32)

output_signature = (
    (tf.TensorSpec(shape=Xtrain.shape[1:], dtype=tf.float32),
     tf.TensorSpec(shape=noise_shape, dtype=tf.float32)),
    tf.TensorSpec(shape=Ytrain.shape[1:], dtype=tf.float32)
)

BATCH_SIZE = 16
train_dataset = tf.data.Dataset.from_generator(data_gen, output_signature=output_signature).batch(BATCH_SIZE)

# Use strategy.scope to distribute the model creation and training
with strategy.scope():
    # Include generator_fit function here (unchanged)
    # [The full definition of generator_fit as already provided should be included here]

    generator_model, training_history = generator_fit(
        train_dataset, Xval, Yval,
        BATCH_SIZE=16, LEARNING_RATE=0.01, DROP_RATE=0,
        NUM_EPOCHS=1000, log_filename="training_log_df_V11.csv"
    )


generator_model.summary()

# Save the trained model outside the function
generator_model.save("generator_df_modelV11.h5")
generator_model.save_weights("generator_df_modelV11_weights.h5")

print("Model saved as generator_df_modelV11.keras")
