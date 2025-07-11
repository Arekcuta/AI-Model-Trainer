import tensorflow as tf
import numpy as np
import sys
import os

# Parse arguments
data_path = None
output_name = None
epochs = 5
num_layers = 2
nodes_per_layer = 10
activation = 'relu'

for i, arg in enumerate(sys.argv):
    if arg == '--data' and i + 1 < len(sys.argv):
        data_path = sys.argv[i + 1]
    if arg == '--output' and i + 1 < len(sys.argv):
        output_name = sys.argv[i + 1]
    if arg == '--epochs' and i + 1 < len(sys.argv):
        epochs = int(sys.argv[i + 1])
    if arg == '--layers' and i + 1 < len(sys.argv):
        num_layers = int(sys.argv[i + 1])
    if arg == '--nodes' and i + 1 < len(sys.argv):
        nodes_per_layer = int(sys.argv[i + 1])
    if arg == '--activation' and i + 1 < len(sys.argv):
        activation = sys.argv[i + 1]

print("TensorFlow version:", tf.__version__)

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


# Dummy fallback data if no --data provided
if data_path:
    print(f"Loading data from {data_path}")
    # For now: assume numpy .npz with 'X' and 'y' arrays
    data = np.load(data_path)
    X = data['X']
    y = data['y']
else:
    print("No data provided, using dummy data.")
    X = np.random.rand(100, 3)
    y = np.random.rand(100, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(3,)))
for _ in range(num_layers):
    model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation))
model.add(tf.keras.layers.Dense(1))


model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=epochs)

if output_name:
    output_path = f"models/{output_name}"
    if not output_path.endswith('.weights.h5'):
        output_path += '.weights.h5'
    print(f"Saving weights to {output_path}")
    os.makedirs("models", exist_ok=True)
    model.save_weights(output_path)
else:
    print("No output name provided, skipping save.")


print("Training complete!")
